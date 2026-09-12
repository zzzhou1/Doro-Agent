"""Token accounting: prompt-cache tokens must count toward input.

Regression cover for "Tokens: 0 in". On a cache hit Anthropic reports
``input_tokens: 0`` and puts the real input in ``cache_read_input_tokens``;
taking that 0 at face value both misreported the total and froze every
context-management threshold, which is keyed off ``last_input_token_count``.

The OpenAI branch has the mirror-image hazard: there ``prompt_tokens`` already
*includes* the cached prefix, so adding it back the way Anthropic needs would
double-count it. Both directions are pinned below.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mini_claude.agent import Agent, _anthropic_usage_totals, _openai_usage_totals
from mini_claude.ui import (
    cache_read_discount_for,
    estimate_cost_usd,
    format_token_usage,
)


class _Usage:
    """Stand-in for anthropic.types.Usage."""

    def __init__(
        self,
        input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


def test_cached_input_counts_as_input() -> None:
    """The exact payload that produced "0 in": everything came from cache."""
    total, cache_write, cache_read = _anthropic_usage_totals(
        _Usage(input_tokens=0, cache_creation_input_tokens=23, cache_read_input_tokens=8814)
    )
    assert total == 8837
    assert (cache_write, cache_read) == (23, 8814)


def test_uncached_input_is_unchanged() -> None:
    assert _anthropic_usage_totals(_Usage(input_tokens=1433)) == (1433, 0, 0)


def test_missing_cache_fields_are_tolerated() -> None:
    """A provider that omits the cache fields must not raise."""

    class Bare:
        input_tokens = 100

    assert _anthropic_usage_totals(Bare()) == (100, 0, 0)


def test_none_values_do_not_leak_into_the_total() -> None:
    """The SDK types these as Optional; None must read as zero, not blow up."""
    usage = _Usage()
    usage.input_tokens = None
    usage.cache_read_input_tokens = None
    assert _anthropic_usage_totals(usage) == (0, 0, 0)


def test_cost_prices_cache_apart_from_fresh_input() -> None:
    # Fresh input: full rate.
    assert estimate_cost_usd(1_000_000, 0) == pytest.approx(3.0)
    # Cache read: 10% of the input rate — the whole point of caching.
    assert estimate_cost_usd(1_000_000, 0, cache_read_tokens=1_000_000) == pytest.approx(0.30)
    # Cache write: 125% of the input rate.
    assert estimate_cost_usd(1_000_000, 0, cache_write_tokens=1_000_000) == pytest.approx(3.75)


def test_cost_of_a_cache_hit_is_not_charged_at_the_fresh_rate() -> None:
    """A fully cached 8837-token request must not bill like fresh input."""
    cached = estimate_cost_usd(8837, 2, cache_read_tokens=8814)
    fresh = estimate_cost_usd(8837, 2)
    assert cached < fresh / 5


def test_format_marks_cached_tokens_only_when_present() -> None:
    assert format_token_usage(100, 5) == "Tokens: 100 in / 5 out"
    assert format_token_usage(8837, 2, cache_read_tokens=8814) == (
        "Tokens: 8837 in (8814 cached) / 2 out"
    )
    assert format_token_usage(8837, 2, cache_read_tokens=8000, cache_write_tokens=814) == (
        "Tokens: 8837 in (8814 cached) / 2 out"
    )


# ─── OpenAI branch ───────────────────────────────────────────


class _OpenAiUsage:
    """Stand-in for openai.types.CompletionUsage."""

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        with_details: bool = True,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if with_details:
            self.prompt_tokens_details = SimpleNamespace(cached_tokens=cached_tokens)


def test_openai_cached_tokens_are_a_subset_not_an_addend() -> None:
    """The regression that copying the Anthropic fix verbatim would create.

    ``prompt_tokens`` is already the whole prompt; the cached prefix only says
    which slice of it to discount. Summing it in would report 4356 for a
    2372-token request and push the context gauge past the real window.
    """
    total, output, cache_read = _openai_usage_totals(
        _OpenAiUsage(prompt_tokens=2372, completion_tokens=8, cached_tokens=1984)
    )
    assert total == 2372
    assert output == 8
    assert cache_read == 1984


def test_openai_uncached_request_reports_no_cache_read() -> None:
    assert _openai_usage_totals(_OpenAiUsage(prompt_tokens=604, completion_tokens=10)) == (
        604,
        10,
        0,
    )


def test_openai_missing_prompt_token_details_is_tolerated() -> None:
    """Gateways that don't implement prompt caching omit the whole details block."""

    class Bare:
        prompt_tokens = 100
        completion_tokens = 5

    assert _openai_usage_totals(Bare()) == (100, 5, 0)


def test_openai_none_values_do_not_leak_into_the_total() -> None:
    usage = _OpenAiUsage(with_details=False)
    usage.prompt_tokens = None
    usage.completion_tokens = None
    assert _openai_usage_totals(usage) == (0, 0, 0)


def test_openai_none_cached_tokens_reads_as_zero() -> None:
    """The SDK types this field Optional; it was observed null on a real gateway."""
    usage = _OpenAiUsage(prompt_tokens=500, completion_tokens=3)
    usage.prompt_tokens_details.cached_tokens = None
    assert _openai_usage_totals(usage) == (500, 3, 0)


def test_openai_cache_read_cannot_exceed_the_prompt_it_discounts() -> None:
    """A gateway reporting cached > prompt must not produce a negative fresh slice."""
    _total, _output, cache_read = _openai_usage_totals(
        _OpenAiUsage(prompt_tokens=100, completion_tokens=1, cached_tokens=5000)
    )
    assert cache_read == 100
    assert estimate_cost_usd(100, 1, cache_read_tokens=cache_read, cache_read_discount=0.5) >= 0


def test_cache_read_discount_is_per_backend() -> None:
    assert cache_read_discount_for("anthropic") == 0.1
    assert cache_read_discount_for("openai") == 0.5
    # Unknown backends fall back to the more conservative Anthropic rate.
    assert cache_read_discount_for("something-else") == 0.1


def test_openai_cache_read_is_priced_at_half_rate_not_tenth() -> None:
    """1M cached tokens: $1.50 on OpenAI's 0.5x, $0.30 on Anthropic's 0.1x.

    Pricing OpenAI at Anthropic's rate would understate spend fivefold.
    """
    kwargs = {"cache_read_tokens": 1_000_000}
    assert estimate_cost_usd(1_000_000, 0, cache_read_discount=0.5, **kwargs) == pytest.approx(1.50)
    assert estimate_cost_usd(1_000_000, 0, cache_read_discount=0.1, **kwargs) == pytest.approx(0.30)


def _chunk(text: str | None = None, finish: str | None = None, usage=None):
    choices = []
    if text is not None or finish is not None:
        choices.append(
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish,
            )
        )
    return SimpleNamespace(choices=choices, usage=usage)


class _FakeOpenAiClient:
    def __init__(self, chunks):
        self._chunks = chunks

        class _Completions:
            async def create(_self, **_kwargs):
                async def _gen():
                    for chunk in chunks:
                        yield chunk

                return _gen()

        self.chat = SimpleNamespace(completions=_Completions())


async def _stream_once(chunks):
    with patch("mini_claude.agent.openai.AsyncOpenAI"), patch(
        "mini_claude.agent.stop_spinner"
    ):
        agent = Agent(backend="openai", api_key="test-key", custom_system_prompt="p")
    agent._openai_client = _FakeOpenAiClient(chunks)
    agent._output_buffer = []  # swallow emitted text instead of printing it
    return await agent._call_openai_stream()


async def test_stream_without_a_usage_chunk_reports_none_not_zero() -> None:
    """A gateway that ignores ``include_usage`` must not zero the context gauge.

    The old code substituted ``{"prompt_tokens": 0, "completion_tokens": 0}``,
    which is truthy, so the caller dutifully set ``last_input_token_count = 0``
    and auto-compaction stopped firing for the rest of the session.
    """
    response = await _stream_once([_chunk(text="pong", finish="stop")])
    assert response["usage"] is None


async def test_stream_keeps_the_nested_cache_details() -> None:
    """The raw usage object is preserved, so the cached count survives."""
    usage = _OpenAiUsage(prompt_tokens=2372, completion_tokens=8, cached_tokens=1984)
    response = await _stream_once(
        [_chunk(text="pong", finish="stop"), _chunk(usage=usage)]
    )
    total, output, cache_read = _openai_usage_totals(response["usage"])
    assert (total, output, cache_read) == (2372, 8, 1984)
    assert response["choices"][0]["message"]["content"] == "pong"
