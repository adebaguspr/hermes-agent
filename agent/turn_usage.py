"""Per-response usage accounting for the conversation turn loop.

After every successful model API call, ``record_response_usage`` folds ``response.usage``
into: the context compressor (``update_from_response`` + the compression-budget rearm
latch), the usage anchor for display/compression math, per-session token/cost counters,
the state.db token-delta queue, and the observability log line. MoA sessions additionally
fold advisor fan-out usage into the reported counts and price the aggregator at its REAL
model/provider. Logger name stays ``agent.conversation_loop`` for caplog parity.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List

from agent.image_token_cost import calibrate_from_usage
from agent.usage_anchor import capture_usage_anchor, set_usage_anchor
from agent.usage_pricing import estimate_usage_cost, normalize_usage, with_served_service_tier

logger = logging.getLogger("agent.conversation_loop")


def _agent_session_source(agent: Any) -> str:
    """The surface the agent's own row create would stamp (``_ensure_db_session``), so an
    accounting guard that wins the row-creation race never mints an anonymous session."""
    from run_agent import _session_source_for_agent  # late: run_agent imports this module
    return _session_source_for_agent(getattr(agent, "platform", None))


@dataclass
class ResponseUsageOutcome:
    """``compression_attempts`` is the (possibly rearmed-to-zero) budget counter;
    ``rearmed`` tells the loop to also clear its preflight-block latch."""

    compression_attempts: int
    rearmed: bool = False


def _loop_mod():
    """Lazy ``agent.conversation_loop`` import (avoids an import cycle)."""
    import agent.conversation_loop as _cl

    return _cl


def _fold_moa_usage(agent, canonical_usage):
    """MoA: fold advisor fan-out usage into REPORTED token counts (only aggregator usage is
    returned, so advisor spend would be invisible) and flush the full-turn trace when
    ``moa.save_traces`` is on. Returns ``(client, canonical_usage, advisor_cost)``."""
    _moa_ref_cost = None
    _moa_client = getattr(agent, "client", None)
    if _moa_client is not None and hasattr(_moa_client, "consume_reference_usage"):
        try:
            _ref_usage, _moa_ref_cost = _moa_client.consume_reference_usage()
            if _ref_usage is not None:
                canonical_usage = canonical_usage + _ref_usage
        except Exception as _moa_acct_exc:  # pragma: no cover - defensive
            logger.debug("MoA reference usage accounting failed: %s", _moa_acct_exc)
    if _moa_client is not None and hasattr(_moa_client, "consume_and_save_trace"):
        try:
            # Streaming path: pass the streamed acting text so the trace is self-contained.
            _agg_streamed_text = getattr(agent, "_current_streamed_assistant_text", "") or ""
            _moa_client.consume_and_save_trace(
                agent.session_id, aggregator_output_fallback=_agg_streamed_text or None
            )
        except Exception as _moa_trace_exc:  # pragma: no cover - defensive
            logger.debug("MoA trace flush failed: %s", _moa_trace_exc)
    return _moa_client, canonical_usage, _moa_ref_cost


def record_response_usage(
    agent: Any, response: Any, *, messages: List[Dict[str, Any]], api_call_count: int,
    api_duration: float, compression_attempts: int, max_compression_attempts: int,
) -> ResponseUsageOutcome:
    """Fold ``response.usage`` into compressor, anchors, session counters, state.db
    and the API-call log line (see module docstring). No-usage responses only
    consume a pending compaction verdict. Returns the loop-visible outcome."""
    rearmed = False
    compressor = agent.context_compressor
    # Count every completed provider attempt, including providers that omit usage.
    # Token/cost accounting below stays gated on real usage, but the request itself
    # must remain observable.
    agent.session_api_calls += 1
    if not (hasattr(response, 'usage') and response.usage):
        if getattr(compressor, "awaiting_real_usage_after_compression", False):
            # No usage -> cannot adjudicate the prior compaction; consume the
            # pending verdict so later readings aren't charged to it and
            # preflight deferral isn't latched indefinitely.
            compressor.update_from_response({})
        _note_usage_less = getattr(compressor, "note_usage_less_response", None)
        if callable(_note_usage_less):
            _note_usage_less()
        logger.info(
            "API call #%d: model=%s provider=%s in=? out=? total=? latency=%.1fs usage=unavailable",
            agent.session_api_calls, agent.model, agent.provider or "unknown", api_duration,
        )
        return ResponseUsageOutcome(compression_attempts=compression_attempts, rearmed=rearmed)

    canonical_usage = with_served_service_tier(
        normalize_usage(response.usage, provider=agent.provider, api_mode=agent.api_mode), response)
    # Aggregator-only usage kept for pricing: advisor tokens are priced at each advisor's
    # OWN model rate and added as dollars below.
    aggregator_usage = canonical_usage
    _moa_client, canonical_usage, _moa_ref_cost = _fold_moa_usage(agent, canonical_usage)
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    # Canonical token + cache buckets for context engines; legacy keys stay for back-compat.
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }
    # Capture the boundary latch before update_from_response() consumes it: only the real
    # prompt count right after a compaction rearms the budget.
    _completed_compaction_pending = bool(
        getattr(compressor, "_verify_compaction_cleared_threshold", False)
    )
    compressor.update_from_response(usage_dict)
    # Usage-anchored accounting: snapshot exact provider usage against the durable
    # transcript (main-loop ONLY; MoA uses pre-fold aggregator usage). The display meter
    # anchors on the turn's FIRST response: later same-turn responses inflate
    # prompt_tokens with replayed thinking. Display-only; compression math uses real usage.
    # The provider just priced this request exactly: if the delta since the previous anchor
    # introduced images, the residual is their real per-image cost (learned before re-anchoring).
    calibrate_from_usage(agent, messages, aggregator_usage.prompt_tokens)
    _new_anchor = capture_usage_anchor(
        aggregator_usage.prompt_tokens, aggregator_usage.output_tokens, messages
    )
    if _new_anchor is not None:
        set_usage_anchor(agent, _new_anchor, turn_base=api_call_count == 1)
    _compression_threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
    if _loop_mod()._should_rearm_compression_budget(
        compression_attempts, completed_compaction_pending=_completed_compaction_pending,
        prompt_tokens=prompt_tokens, threshold_tokens=_compression_threshold,
    ):
        logger.info(
            "Compression budget rearmed after provider-confirmed "
            "recovery: prompt=%s < threshold=%s (attempts were %s/%s)",
            f"{prompt_tokens:,}",
            f"{_compression_threshold:,}",
            compression_attempts,
            max_compression_attempts,
        )
        compression_attempts = 0
        # Confirmed recovery also clears the loop's stale insufficient-progress verdict
        # (``_preflight_compression_blocked``), else a later pressure spike grows unchecked.
        rearmed = True

    # Stash canonical usage for on_turn_complete(); keep the latest call's.
    agent._last_turn_usage = dict(usage_dict)
    # The parent's CURRENT prompt size for headroom math (delegate summary budgets): the
    # aggregator's own prompt, never the MoA-folded total (advisor prompts are not in this context).
    agent._last_prompt_size_tokens = int(aggregator_usage.prompt_tokens or 0)

    # Persist only provider-confirmed context lengths, not probe tiers.
    if getattr(compressor, "_context_probed", False):
        ctx = compressor.context_length
        if getattr(compressor, "_context_probe_persistable", False):
            from agent.model_metadata import save_provider_context_length

            save_provider_context_length(agent.model, agent.base_url, ctx, agent.provider)
            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
        compressor._context_probed = False
        compressor._context_probe_persistable = False

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens
    # Rolling history for status-bar averages (last 10).
    with suppress(Exception):
        hist = getattr(agent, "_api_latency_history", None)
        if hist is not None:
            hist.append(float(api_duration))
        ohist = getattr(agent, "_api_output_history", None)
        if ohist is not None:
            ohist.append(int(canonical_usage.output_tokens or 0))

    _cache_pct = ""
    if canonical_usage.cache_read_tokens and prompt_tokens:
        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
    # write= is the money (cache writes cost 50x a read); id= is what a provider needs to look the
    # request up; upstream= is who actually served it when the route reports that (OpenRouter's
    # `provider`). Diagnosing the 1,393-agent run's cache misses took a DB join and a live probe
    # because none of the three were on this line.
    if canonical_usage.cache_write_tokens:
        _cache_pct += f" write={canonical_usage.cache_write_tokens}"
    _rid = getattr(response, "id", None)
    _ident = f" id={_rid}" if isinstance(_rid, str) and _rid else ""
    _upstream = getattr(response, "provider", None)
    if isinstance(_upstream, str) and _upstream:
        _ident += f" upstream={_upstream}"
    logger.info(
        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s%s",
        agent.session_api_calls, agent.model, agent.provider or "unknown",
        prompt_tokens, completion_tokens, total_tokens,
        api_duration, _cache_pct, _ident,
    )
    # nous.anthropic_wire=auto: the session's wire is decided once, from this first response.
    if agent.session_api_calls == 1 and (agent.provider or "") == "nous":
        with suppress(Exception):
            from agent.nous_wire import maybe_switch_wire_after_first_response
            maybe_switch_wire_after_first_response(agent, response, agent.session_api_calls)

    # MoA: agent.model/provider are the virtual preset/"moa" with no pricing entry, silently
    # dropping aggregator spend. Price at the REAL model/provider from the aggregator slot.
    _agg_cost_model, _agg_cost_provider, _agg_cost_base_url = agent.model, agent.provider, agent.base_url
    _agg_slot = getattr(_moa_client, "last_aggregator_slot", None) if _moa_client is not None else None
    if _agg_slot and _agg_slot.get("model"):
        _agg_cost_model = _agg_slot["model"]
        _agg_cost_provider = _agg_slot.get("provider") or agent.provider
        _agg_cost_base_url = _agg_slot.get("base_url") or agent.base_url
    cost_result = estimate_usage_cost(
        _agg_cost_model, aggregator_usage, provider=_agg_cost_provider,
        base_url=_agg_cost_base_url, api_key=getattr(agent, "api_key", ""),
    )
    # Cost delta = aggregator + MoA advisor cost (already priced per-advisor at each
    # advisor's own model rate), so state.db's estimated_cost_usd matches the folded
    # token counts.
    _cost_delta = None
    if cost_result.amount_usd is not None:
        _cost_delta = float(cost_result.amount_usd)
        agent.session_estimated_cost_usd += _cost_delta
    if _moa_ref_cost is not None:
        try:
            _moa_cost = float(_moa_ref_cost)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            _moa_cost = None
        if _moa_cost is not None:
            agent.session_estimated_cost_usd += _moa_cost
            _cost_delta = (_cost_delta or 0.0) + _moa_cost
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    # Persist per-call token deltas for any session_id so non-CLI runs can't lose
    # accounting; gateway/session-store writes use absolute totals and safely overwrite
    # these deltas. Enqueued, not written (a cold state.db UPDATE here stalled the tool
    # loop); drained at finalize via _persist_session.
    if agent._session_db and agent.session_id:
        try:
            # Ensure the row exists: under concurrent SQLite load the initial
            # _ensure_db_session() may fail, and UPDATE on a missing row affects 0 rows.
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.queue_token_counts(
                agent.session_id,
                source=_agent_session_source(agent),
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=_cost_delta,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as e:  # silent loss here undercounts analytics
            logger.debug(
                "Token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, e,
            )

    if agent.verbose_logging:
        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")

    # Report cache stats for any provider that returns ``prompt_tokens_details.cached_tokens``,
    # not only when we inject cache_control markers.
    cached = canonical_usage.cache_read_tokens
    written = canonical_usage.cache_write_tokens
    prompt = usage_dict["prompt_tokens"]
    if (cached or written) and not agent.quiet_mode:
        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
        agent._vprint(
            f"{agent.log_prefix}   💾 Cache: "
            f"{cached:,}/{prompt:,} tokens "
            f"({hit_pct:.0f}% hit, {written:,} written)"
        )

    _record_token_ledger(agent, response, canonical_usage, messages)

    return ResponseUsageOutcome(compression_attempts=compression_attempts, rearmed=rearmed)


def _record_token_ledger(
    agent: Any,
    response: Any,
    canonical_usage: Any,
    messages: List[Dict[str, Any]],
) -> None:
    """Durable append-only token ledger for observability and economics audits.
    Never raises exceptions."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        telemetry_cfg = cfg.get("telemetry") or {}
        if not telemetry_cfg.get("token_ledger", True):
            return
    except Exception:
        # If config is unreachable, fail-open (leave ledger active)
        pass

    try:
        import json
        import os
        from hermes_time import now as _hermes_now

        logs_dir = os.path.expanduser("~/.hermes/logs")
        os.makedirs(logs_dir, exist_ok=True)

        local_now = _hermes_now()
        date_str = local_now.strftime("%Y-%m-%d")
        ledger_file = os.path.join(logs_dir, f"token-ledger-{date_str}.jsonl")

        # Determine delegation role / id
        deleg_val = "main"
        if getattr(agent, "is_subagent", False) or getattr(agent, "_delegate_depth", 0) > 0:
            deleg_val = str(getattr(agent, "_subagent_id", "") or getattr(agent, "subagent_id", "") or "subagent")
        elif hasattr(agent, "platform") and agent.platform == "subagent":
            deleg_val = str(getattr(agent, "_subagent_id", "") or getattr(agent, "subagent_id", "") or "subagent")

        # Inspect raw response usage for cache stats
        raw_u = getattr(response, "usage", None)
        raw_dict = {}
        if isinstance(raw_u, dict):
            raw_dict = raw_u
        elif hasattr(raw_u, "model_dump") and callable(raw_u.model_dump):
            raw_dict = raw_u.model_dump()
        elif hasattr(raw_u, "__dict__"):
            raw_dict = vars(raw_u)

        pt_details = raw_dict.get("prompt_tokens_details") or {}
        if not isinstance(pt_details, dict) and hasattr(pt_details, "__dict__"):
            pt_details = vars(pt_details)

        cached_field_present = False
        cached_tokens_val = 0

        if isinstance(pt_details, dict) and "cached_tokens" in pt_details:
            cached_field_present = True
            cached_tokens_val = int(pt_details.get("cached_tokens") or 0)
        elif "cached_tokens" in raw_dict:
            cached_field_present = True
            cached_tokens_val = int(raw_dict.get("cached_tokens") or 0)
        elif "cache_read_input_tokens" in raw_dict:
            cached_field_present = True
            cached_tokens_val = int(raw_dict.get("cache_read_input_tokens") or 0)
        elif getattr(canonical_usage, "cache_read_tokens", 0):
            cached_field_present = True
            cached_tokens_val = int(canonical_usage.cache_read_tokens)

        cache_read_input_tokens_val = int(
            raw_dict.get("cache_read_input_tokens")
            or (pt_details.get("cached_tokens") if isinstance(pt_details, dict) else 0)
            or getattr(canonical_usage, "cache_read_tokens", 0)
            or 0
        )
        cache_creation_input_tokens_val = int(
            raw_dict.get("cache_creation_input_tokens")
            or getattr(canonical_usage, "cache_write_tokens", 0)
            or 0
        )

        system_chars = 0
        sys_prompt = getattr(agent, "_cached_system_prompt", None)
        if isinstance(sys_prompt, str):
            system_chars = len(sys_prompt)

        tools_chars = 0
        tools = getattr(agent, "tools", None)
        if tools:
            try:
                tools_chars = len(json.dumps(tools))
            except Exception:
                pass

        history_chars = 0
        if messages:
            try:
                history_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
            except Exception:
                pass

        ledger_entry = {
            "ts": local_now.isoformat(),
            "model": str(getattr(agent, "model", "") or ""),
            "deleg": deleg_val,
            "prompt_tokens": int(getattr(canonical_usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(canonical_usage, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(canonical_usage, "total_tokens", 0) or 0),
            "cached_tokens": cached_tokens_val,
            "cached_field_present": cached_field_present,
            "cache_read_input_tokens": cache_read_input_tokens_val,
            "cache_creation_input_tokens": cache_creation_input_tokens_val,
            "system_chars": system_chars,
            "tools_chars": tools_chars,
            "history_chars": history_chars,
        }

        with open(ledger_file, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(ledger_entry) + chr(10))
    except Exception:
        pass
