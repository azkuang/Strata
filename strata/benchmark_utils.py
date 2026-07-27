"""Normalizes M2/M3/M4's differently-shaped generation results
(BatchGenerationResult, ContinuousBatchGenerationResult -- see
strata/engine.py) plus a ResourceReport into one flat result row for
scripts/benchmark_m5.py to append to a JSONL file.
"""

from __future__ import annotations

from typing import Any

from strata.resource_monitor import ResourceReport


def total_generated_tokens(token_ids: list[list[int]]) -> int:
    """Total decode-phase tokens across all sequences: each sequence's
    prefill call already contributes its first token (see engine.py's
    admit()/prefill methods), so only len(seq) - 1 per sequence counts as
    decode -- matching every engine's own decode_tok_per_s convention.
    """
    return sum(max(len(seq) - 1, 0) for seq in token_ids)


def build_result_row(
    *,
    engine: str,
    concurrency: int,
    max_new_tokens: int,
    token_ids: list[list[int]],
    engine_decode_tok_per_s: float,
    ttft_ms: float | None,
    total_global_steps: int | None,
    resource_report: ResourceReport,
) -> dict[str, Any]:
    """Assemble one JSONL-ready benchmark result row.

    uniform_tok_per_s (total decode tokens / resource_report.duration_s,
    i.e. the whole generate() call including prefill/admission) is the
    fair cross-engine comparison figure -- M2's own decode_tok_per_s
    excludes its prefill phase while M3/M4's own decode_tok_per_s divides
    by their full wall clock (see README.md's Benchmarks section, which
    already flags M2 vs M3's throughput numbers as not strictly
    like-for-like for this reason). engine_decode_tok_per_s is kept
    alongside it for continuity with each milestone's own prior
    self-reported figure.

    ttft_ms is only meaningful for M2 (BatchEngine), which has one shared
    prefill phase; M3/M4 admit sequences at different wall-clock times, so
    there's no single TTFT -- pass None for those (per this plan's Global
    Constraints) and use approx_itl_ms (resource_report.duration_s /
    total_global_steps) instead.
    """
    total_tokens = total_generated_tokens(token_ids)
    duration_s = resource_report.duration_s
    uniform_tok_per_s = total_tokens / duration_s if duration_s > 0 else 0.0

    approx_itl_ms = None
    if total_global_steps:
        approx_itl_ms = (duration_s / total_global_steps) * 1000

    tok_per_s_per_w = None
    if resource_report.avg_power_w:
        tok_per_s_per_w = uniform_tok_per_s / resource_report.avg_power_w

    return {
        "engine": engine,
        "concurrency": concurrency,
        "max_new_tokens": max_new_tokens,
        "wall_clock_s": duration_s,
        "total_tokens": total_tokens,
        "uniform_tok_per_s": uniform_tok_per_s,
        "engine_decode_tok_per_s": engine_decode_tok_per_s,
        "ttft_ms": ttft_ms,
        "approx_itl_ms": approx_itl_ms,
        "total_global_steps": total_global_steps,
        "avg_power_w": resource_report.avg_power_w,
        "num_power_samples": resource_report.num_power_samples,
        "tok_per_s_per_w": tok_per_s_per_w,
        "peak_allocated_gb": resource_report.peak_allocated_gb,
        "peak_reserved_gb": resource_report.peak_reserved_gb,
        "peak_rss_gb": resource_report.peak_rss_gb,
    }
