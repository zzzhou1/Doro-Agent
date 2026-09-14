"""Per-model pricing and cost estimation.

The old code hardcoded one Anthropic-style rate card ($3/$15 per million) for
every model on every backend, so the cost line was structurally wrong for a
gateway model and silently mis-priced cache reads (Anthropic discounts them
0.1x, OpenAI-compatible endpoints 0.5x).

Two rules keep this honest:

1. ``PRICING_TABLE`` only holds prices that are publicly documented first-party
   list prices. Gateway aliases (``gpt-5.6-sol``, ``claude-opus-5``, ...) are
   deliberately *absent* — inventing a number for them would make an estimate
   look like a measured fact.
2. When no price is known the estimate falls back to ``DEFAULT_PRICING`` and is
   flagged ``estimated=True`` so every surface that shows it can say so.

Prices can be overridden from ``.env``:

    DORO_PRICES={"gateway-model": {"input": 2, "output": 8}}
    DORO_PRICE_INPUT=2          # per-million USD, applies to every model
    DORO_PRICE_OUTPUT=8
    DORO_PRICE_CACHE_WRITE=2.5
    DORO_PRICE_CACHE_READ=0.4
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens for one model."""

    input_per_m: float
    output_per_m: float
    cache_write_per_m: float
    cache_read_per_m: float


# Cache reads are billed as a fraction of the input rate, and the fraction is a
# property of the provider, not of the model.
CACHE_READ_DISCOUNT: dict[str, float] = {"anthropic": 0.1, "openai": 0.5}
DEFAULT_CACHE_READ_DISCOUNT = 0.1


def cache_read_discount_for(backend: str) -> float:
    """Cache-read multiplier for a backend id (``openai``/``anthropic``)."""
    return CACHE_READ_DISCOUNT.get(backend, DEFAULT_CACHE_READ_DISCOUNT)


# Fallback rate card: Anthropic Sonnet-tier list price, with the cache-read rate
# derived from the backend's discount so an unknown model on the OpenAI backend
# is not under-priced fivefold. Used only when nothing better is known.
DEFAULT_PRICING = ModelPricing(
    input_per_m=3.0,
    output_per_m=15.0,
    cache_write_per_m=3.75,
    cache_read_per_m=3.0 * DEFAULT_CACHE_READ_DISCOUNT,
)

# Published list prices, USD per million tokens, longest prefix wins. Anthropic
# cache-write is 1.25x input and cache-read is 0.1x input; OpenAI-compatible
# endpoints charge no cache-write surcharge (writes bill as ordinary input) and
# discount cached reads to the documented ``cached_tokens`` rate.
PRICING_TABLE: dict[str, ModelPricing] = {
    # ── Anthropic ──
    "claude-opus-4": ModelPricing(15.0, 75.0, 18.75, 1.50),
    "claude-sonnet-4": ModelPricing(3.0, 15.0, 3.75, 0.30),
    "claude-haiku-4-5": ModelPricing(1.0, 5.0, 1.25, 0.10),
    "claude-3-5-sonnet": ModelPricing(3.0, 15.0, 3.75, 0.30),
    "claude-3-5-haiku": ModelPricing(0.80, 4.0, 1.00, 0.08),
    # ── OpenAI ──
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.15, 0.075),
    "gpt-4o": ModelPricing(2.50, 10.0, 2.50, 1.25),
    "gpt-4.1-nano": ModelPricing(0.10, 0.40, 0.10, 0.025),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60, 0.40, 0.10),
    "gpt-4.1": ModelPricing(2.00, 8.00, 2.00, 0.50),
}

PRICES_JSON_ENV = "DORO_PRICES"
PRICE_FIELD_ENV: dict[str, str] = {
    "input": "DORO_PRICE_INPUT",
    "output": "DORO_PRICE_OUTPUT",
    "cache_write": "DORO_PRICE_CACHE_WRITE",
    "cache_read": "DORO_PRICE_CACHE_READ",
}


@dataclass(frozen=True)
class PriceResolution:
    """Which rate card applies to a model, and where it came from."""

    model: str
    pricing: ModelPricing
    source: str
    estimated: bool
    match: str  # "env" | "exact" | "family" | "fallback"

    def summary(self) -> str:
        p = self.pricing
        return (
            f"${p.input_per_m:g}/M in · ${p.output_per_m:g}/M out · "
            f"${p.cache_write_per_m:g}/M cache-write · "
            f"${p.cache_read_per_m:g}/M cache-read"
        )


def _fallback_pricing(backend: str | None) -> ModelPricing:
    return replace(
        DEFAULT_PRICING,
        cache_read_per_m=(
            DEFAULT_PRICING.input_per_m
            * cache_read_discount_for(backend or "anthropic")
        ),
    )


def _table_lookup(model: str) -> tuple[ModelPricing | None, str, str]:
    """Longest-prefix lookup. Returns ``(pricing, match, source)``."""
    key = model.strip().lower()
    if not key:
        return None, "fallback", ""
    if key in PRICING_TABLE:
        return PRICING_TABLE[key], "exact", "built-in price table"
    best_prefix = ""
    for prefix in PRICING_TABLE:
        if key.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
    if best_prefix:
        return (
            PRICING_TABLE[best_prefix],
            "family",
            f"built-in price table (family {best_prefix})",
        )
    return None, "fallback", ""


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _env_model_overrides(
    model: str, env: Mapping[str, str]
) -> tuple[dict[str, float], str]:
    """Per-model overrides from ``DORO_PRICES`` (a JSON object)."""
    raw = env.get(PRICES_JSON_ENV)
    if not raw:
        return {}, ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}, ""
    if not isinstance(parsed, dict):
        return {}, ""
    entry = parsed.get(model) or parsed.get(model.strip().lower())
    if not isinstance(entry, dict):
        return {}, ""
    values: dict[str, float] = {}
    for field in PRICE_FIELD_ENV:
        number = _positive_float(entry.get(field))
        if number is not None:
            values[field] = number
    return (values, PRICES_JSON_ENV) if values else ({}, "")


def _env_field_overrides(
    env: Mapping[str, str]
) -> tuple[dict[str, float], str]:
    """Global per-field overrides from ``DORO_PRICE_*``."""
    values: dict[str, float] = {}
    for field, name in PRICE_FIELD_ENV.items():
        number = _positive_float(env.get(name))
        if number is not None:
            values[field] = number
    return values, "DORO_PRICE_*"


def resolve_pricing(
    model: str,
    *,
    backend: str | None = None,
    env: Mapping[str, str] | None = None,
) -> PriceResolution:
    """Resolve the rate card for ``model``.

    Precedence: ``.env`` per-model map > ``.env`` per-field globals > built-in
    table > fallback. Only the fallback is flagged ``estimated=True`` — an
    explicit override is the operator's number, so it is not an estimate.
    """
    env = os.environ if env is None else env

    base, match, source = _table_lookup(model)
    if base is None:
        base = _fallback_pricing(backend)
        source = "fallback rate card — model price unknown"
        estimated = True
    else:
        estimated = False

    model_values, model_source = _env_model_overrides(model, env)
    field_values, field_source = _env_field_overrides(env)

    values: dict[str, float] = {}
    origin = source
    if field_values:
        values.update(field_values)
        origin = f".env ({field_source})"
    if model_values:
        values.update(model_values)
        origin = f".env ({model_source}, model override)"
    if values:
        base = replace(base, **{f"{field}_per_m": v for field, v in values.items()})
        estimated = False
        match = "env"

    return PriceResolution(
        model=model,
        pricing=base,
        source=origin,
        estimated=estimated,
        match=match,
    )


def compute_cost(
    pricing: ModelPricing,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Estimated USD spend.

    ``input_tokens`` is the *total* input including the cached portion, mirroring
    ``ui.estimate_cost_usd``: the cached slices are subtracted out and re-priced
    at their own rates rather than being billed at the fresh-input rate.
    """
    fresh = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
    return (
        fresh * pricing.input_per_m
        + cache_write_tokens * pricing.cache_write_per_m
        + cache_read_tokens * pricing.cache_read_per_m
        + output_tokens * pricing.output_per_m
    ) / 1_000_000
