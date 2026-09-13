"""Per-model pricing: table lookup, env overrides, and the estimated flag.

The rule these tests protect: a model whose price is not publicly known must
never be presented as if its cost were measured. It falls back to a documented
rate card *and* is flagged estimated.
"""

from __future__ import annotations

import pytest

from mini_claude.pricing import (
    DEFAULT_PRICING,
    PRICING_TABLE,
    ModelPricing,
    compute_cost,
    resolve_pricing,
)


def test_unknown_model_falls_back_and_is_flagged_estimated() -> None:
    resolution = resolve_pricing("gpt-5.6-sol", backend="openai", env={})

    assert resolution.match == "fallback"
    assert resolution.estimated is True
    assert resolution.pricing.input_per_m == DEFAULT_PRICING.input_per_m
    # Cache-read follows the backend discount, so an OpenAI-compatible model is
    # not priced at Anthropic's 0.1x and understated fivefold.
    assert resolution.pricing.cache_read_per_m == pytest.approx(1.5)


def test_anthropic_fallback_uses_the_tenth_rate() -> None:
    resolution = resolve_pricing("claude-opus-5", backend="anthropic", env={})

    assert resolution.estimated is True
    assert resolution.pricing.cache_read_per_m == pytest.approx(0.3)


def test_gateway_aliases_are_not_in_the_table() -> None:
    """Listing a price for a proxied alias would fake precision."""
    for alias in ("gpt-5.6-sol", "claude-opus-5"):
        assert alias not in PRICING_TABLE


def test_exact_match_is_not_estimated() -> None:
    resolution = resolve_pricing("gpt-4o", backend="openai", env={})

    assert resolution.match == "exact"
    assert resolution.estimated is False
    assert resolution.pricing.output_per_m == pytest.approx(10.0)


def test_longest_prefix_wins_for_versioned_ids() -> None:
    """``gpt-4o`` is a prefix of ``gpt-4o-mini`` — the longer entry must win."""
    resolution = resolve_pricing("gpt-4o-mini-2024-07-18", backend="openai", env={})

    assert resolution.pricing.output_per_m == pytest.approx(0.60)
    assert resolution.pricing.input_per_m == pytest.approx(0.15)


def test_dated_anthropic_id_matches_its_family() -> None:
    resolution = resolve_pricing("claude-sonnet-4-20250514", backend="anthropic", env={})

    assert resolution.match == "family"
    assert resolution.estimated is False
    assert resolution.pricing.input_per_m == pytest.approx(3.0)


def test_empty_model_name_falls_back() -> None:
    assert resolve_pricing("", backend="openai", env={}).match == "fallback"


# ─── .env overrides ─────────────────────────────────────────


def test_env_json_overrides_one_model() -> None:
    env = {"MINI_CLAUDE_PRICES": '{"gpt-5.6-sol": {"input": 1, "output": 4}}'}
    resolution = resolve_pricing("gpt-5.6-sol", backend="openai", env=env)

    assert resolution.match == "env"
    assert resolution.estimated is False
    assert resolution.pricing.input_per_m == 1.0
    assert resolution.pricing.output_per_m == 4.0
    # Fields the override does not mention keep the underlying card.
    assert resolution.pricing.cache_write_per_m == DEFAULT_PRICING.cache_write_per_m


def test_env_field_vars_apply_to_every_model() -> None:
    env = {"MINI_CLAUDE_PRICE_INPUT": "2", "MINI_CLAUDE_PRICE_CACHE_READ": "0.25"}
    resolution = resolve_pricing("whatever", backend="openai", env=env)

    assert resolution.match == "env"
    assert resolution.pricing.input_per_m == 2.0
    assert resolution.pricing.cache_read_per_m == 0.25


def test_env_json_beats_global_field_vars() -> None:
    env = {
        "MINI_CLAUDE_PRICES": '{"m": {"input": 9}}',
        "MINI_CLAUDE_PRICE_INPUT": "2",
    }
    resolution = resolve_pricing("m", backend="openai", env=env)

    assert resolution.pricing.input_per_m == 9.0


def test_overridden_pricing_is_no_longer_an_estimate() -> None:
    """An explicit operator number is not an approximation."""
    resolution = resolve_pricing(
        "mystery", backend="openai", env={"MINI_CLAUDE_PRICE_INPUT": "1"}
    )
    assert resolution.estimated is False


@pytest.mark.parametrize("raw", ["not json at all", "[]", '{"m": "oops"}', "{}"])
def test_malformed_env_prices_are_ignored(raw: str) -> None:
    resolution = resolve_pricing("m", backend="openai", env={"MINI_CLAUDE_PRICES": raw})
    assert resolution.match == "fallback"


def test_negative_or_non_numeric_env_prices_are_ignored() -> None:
    assert resolve_pricing("m", env={"MINI_CLAUDE_PRICE_INPUT": "-3"}).match == "fallback"
    assert resolve_pricing("m", env={"MINI_CLAUDE_PRICE_OUTPUT": "abc"}).match == "fallback"


# ─── compute_cost ───────────────────────────────────────────


def test_compute_cost_prices_each_slice_at_its_own_rate() -> None:
    pricing = ModelPricing(3.0, 15.0, 3.75, 0.30)

    assert compute_cost(pricing, 1_000_000, 0) == pytest.approx(3.0)
    assert compute_cost(pricing, 0, 1_000_000) == pytest.approx(15.0)
    assert compute_cost(pricing, 1_000_000, 0, cache_read_tokens=1_000_000) == pytest.approx(0.30)
    assert compute_cost(pricing, 1_000_000, 0, cache_write_tokens=1_000_000) == pytest.approx(3.75)


def test_cached_slice_is_not_billed_twice() -> None:
    pricing = ModelPricing(3.0, 15.0, 3.75, 0.30)

    # 100 input total, 90 of it served from cache.
    expected = (10 * 3.0 + 90 * 0.30) / 1_000_000
    assert compute_cost(pricing, 100, 0, cache_read_tokens=90) == pytest.approx(expected)


def test_overflowing_cache_counts_cannot_make_fresh_input_negative() -> None:
    pricing = ModelPricing(3.0, 15.0, 3.75, 0.30)
    assert compute_cost(pricing, 10, 0, cache_read_tokens=99, cache_write_tokens=99) >= 0


def test_summary_is_human_readable() -> None:
    text = resolve_pricing("gpt-4o", backend="openai", env={}).summary()
    assert "in" in text and "out" in text and "cache-read" in text
