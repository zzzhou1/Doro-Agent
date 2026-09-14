"""Shared reasoning-effort configuration for both API backends."""

from __future__ import annotations

REASONING_EFFORTS: tuple[str, ...] = (
    "auto",
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
DEFAULT_REASONING_EFFORT = "medium"
BACKEND_REASONING_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_EFFORT",
    "openai": "OPENAI_REASONING_EFFORT",
}


def normalize_reasoning_effort(value: str) -> str:
    """Return a canonical effort name or raise a user-facing error."""
    normalized = value.strip().lower()
    # ``none`` is the OpenAI wire value. Accept it in environment variables,
    # but keep the user-facing vocabulary backend-neutral.
    if normalized == "none":
        normalized = "off"
    if normalized not in REASONING_EFFORTS:
        choices = ", ".join(REASONING_EFFORTS)
        raise ValueError(
            f"Unknown reasoning effort {value!r}. Choose one of: {choices}."
        )
    return normalized
