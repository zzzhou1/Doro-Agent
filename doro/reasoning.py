"""Shared reasoning-effort configuration for both API backends."""

from __future__ import annotations

REASONING_EFFORTS: tuple[str, ...] = (
    "auto",
    "off",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
DEFAULT_REASONING_EFFORT = "medium"
REASONING_FALLBACKS: dict[str, str] = {
    "max": "xhigh",
    "xhigh": "high",
    "high": "medium",
    "medium": "low",
    "low": "auto",
}
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


def reasoning_fallback_for(error: Exception, current: str) -> str | None:
    """Return the next effort for an explicit reasoning compatibility error.

    Only HTTP 400/422 responses that name a reasoning field qualify. Other
    failures retain their original behavior and are never disguised as model
    capability problems.
    """
    if current in ("auto", "off"):
        return None

    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status not in (400, 422, "400", "422"):
        return None

    message = str(error).lower()
    fields = ("reasoning_effort", "reasoning effort", "output_config", "thinking")
    if not any(field in message for field in fields):
        return None

    field_errors = (
        "unknown parameter",
        "unrecognized parameter",
        "unsupported parameter",
        "unexpected keyword",
        "extra inputs are not permitted",
        "not a permitted parameter",
    )
    if any(marker in message for marker in field_errors):
        return "auto"

    compatibility_errors = (
        "unsupported",
        "not supported",
        "invalid value",
        "unsupported_value",
        "must be one of",
        "expected one of",
        "not available",
    )
    if any(marker in message for marker in compatibility_errors):
        # Doro supports only Anthropic adaptive thinking. If that protocol is
        # rejected, changing its effort value cannot make it compatible.
        if "thinking" in message and "adaptive" in message:
            return "auto"
        return REASONING_FALLBACKS.get(current)
    return None
