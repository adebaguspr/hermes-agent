"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd]}``
(order shown; drop any to hide), per-platform override ``display.platforms.<p>.runtime_footer``,
toggled by ``/footer on|off``. Fields: ``model`` (vendor prefix dropped), ``context_pct`` (last-call
occupancy), ``latency`` (turn wall-clock, opt-in — NOT in the default set so an unset ``fields``
renders exactly as before), ``served_model`` (opt-in, ``alias → served``: the deployment a routing
proxy reported via ``x-litellm-model-id`` / ``x-litellm-model-api-base``, or Hermes' own fallback
route; skipped when the served model is the requested one), ``cwd`` (home-relative). ``gateway/run.py`` appends the footer to the
final response only (never to tool-progress or streaming partials); when streaming already
delivered the text, it goes out as a trailing message via ``send_trailing_footer()``."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    return model.rsplit("/", 1)[-1] if model else ""


def _env_cwd() -> str:
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def resolve_footer_config(user_config: dict[str, Any] | None, platform_key: str | None = None) -> dict[str, Any]:
    """Resolve effective footer config: defaults (enabled=False) <
    ``display.runtime_footer`` < ``display.platforms.<platform_key>.runtime_footer``."""
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}
    plat_cfg = (cfg.get("platforms") or {}).get(platform_key) if platform_key else None
    sections = [cfg.get("runtime_footer"), plat_cfg.get("runtime_footer") if isinstance(plat_cfg, dict) else None]
    for section in sections:
        if not isinstance(section, dict):
            continue
        if "enabled" in section:
            resolved["enabled"] = bool(section.get("enabled"))
        if isinstance(section.get("fields"), list) and section["fields"]:
            resolved["fields"] = [str(f) for f in section["fields"]]
    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def _format_ctx_usage(used: int, total: Optional[int]) -> str:
    """Compact usage: ``12k/200k``, ``87k/1M``. Skipped when total unknown."""
    if not total or total <= 0 or used < 0:
        return ""

    def short(n: int) -> str:
        # Tiered rounding: >=1M whole "M" (integer math, exact), >=10k whole "k"
        # (integer math, exact), >=1k one decimal "k" via :.1f, but drop a redundant
        # ".0" tail so exact multiples render "2k"/"10k" not "2.0k"/"10.0k". <1k raw.
        # Never clamp against total.
        if n >= 1_000_000:
            return f"{(n + 500_000) // 1_000_000}M"
        if n >= 10_000:
            return f"{(n + 500) // 1_000}k"
        if n >= 1_000:
            s = f"{n / 1_000:.1f}"
            if s.endswith(".0"):
                s = s[:-2]
            return s + "k"
        return str(n)

    return f"{short(used)}/{short(total)}"


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          requested_model: Optional[str] = None, served_model: Optional[str] = None,
                          cached_tokens: Optional[int] = None,
                          fields: Iterable[str] = _DEFAULT_FIELDS) -> str:
    """Render the footer line, or "" if no fields have data. Fields whose data is missing (and
    unknown field names) are skipped silently — a partial footer beats ``?%`` or empty slots.

    Exception: ``cache_hit`` and ``ctx_usage`` ALWAYS render (``CH: 0%`` / ``0k/200k``) so a
    zero/unknown reading is visible instead of vanishing — the absence was mistaken for a bug."""
    def context_pct() -> str:
        if context_length and context_length > 0 and context_tokens >= 0:
            return f"{max(0, min(100, round((context_tokens / context_length) * 100)))}%"
        return ""

    def cache_hit() -> str:
        # A zero-read regime hides the segment: "no cache data" must never be rendered as an
        # alarming "CH: 0%". Two cases reach here with a zero numerator — (a) the provider does
        # not report cache at all (9r.LOW returns no cached_tokens), (b) the call was a genuine
        # cold start. Neither is a cache FAILURE, and a red 0% every other turn reads as one.
        # Mirrors the CLI status bar contract (hermes_cli/cli_status_bar_mixin.py): show the
        # ratio only when both sides carry data. Callers pass the PER-CALL pair, so the number
        # reflects the current cache regime rather than a lifetime average.
        hit = max(0, cached_tokens or 0)
        total = context_tokens if context_tokens and context_tokens > 0 else 0
        if not total or not hit:
            return ""
        return f"CH: {max(0, min(100, round(hit / total * 100)))}%"

    def ctx_usage() -> str:
        return _format_ctx_usage(max(0, context_tokens or 0), context_length)

    def served() -> str:
        requested = requested_model or model
        alias = _model_short(requested)
        if served_model and served_model not in (alias, requested):
            return f"{alias} → {served_model}"
        return ""

    renderers = {
        "model": lambda: _model_short(model),
        "served_model": served,
        "context_pct": context_pct,
        "cache_hit": cache_hit,
        "ctx_usage": ctx_usage,
        "ctx": lambda: _format_ctx_usage(context_tokens, context_length),
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
    }
    # ``ctx_usage`` always renders (even "0k/200k") so the context meter never vanishes; a
    # zero READ on ``cache_hit`` is different — it means "no cache data this call", and forcing
    # "CH: 0%" back in would repaint the very alarming reading the renderer just suppressed.
    always = {"ctx_usage"}
    return _SEP.join(v for field in fields if (render := renderers.get(field))
                     and (v := render() or "")
                     and (v or field in always))


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None,
                      requested_model: Optional[str] = None, served_model: Optional[str] = None,
                      cached_tokens: Optional[int] = None) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(model=model, context_tokens=context_tokens,
                                 context_length=context_length, cwd=cwd, turn_seconds=turn_seconds,
                                 requested_model=requested_model, served_model=served_model,
                                 cached_tokens=cached_tokens,
                                 fields=cfg.get("fields") or _DEFAULT_FIELDS)
