"""Fast (non-GPU) tests for M5's result-row normalization: constructs
fake ResourceReport instances directly (no GPU/model needed) to exercise
build_result_row's arithmetic and the token_ids -> uniform tok/s
convention shared by M2/M3/M4's engines (prefill contributes each
sequence's first token, see strata/engine.py's admit()/prefill methods).
"""

import pytest

from strata.benchmark_utils import build_result_row, total_generated_tokens
from strata.resource_monitor import ResourceReport


def _report(**overrides) -> ResourceReport:
    defaults = dict(
        duration_s=10.0,
        avg_power_w=20.0,
        num_power_samples=20,
        peak_allocated_gb=5.0,
        peak_reserved_gb=6.0,
        peak_rss_gb=1.0,
    )
    defaults.update(overrides)
    return ResourceReport(**defaults)


def test_total_generated_tokens_excludes_prefill_token_per_sequence():
    # 3 sequences: 4 tokens (3 decode), 1 token (0 decode), 0 tokens (0 decode)
    assert total_generated_tokens([[1, 2, 3, 4], [1], []]) == 3


def test_build_result_row_computes_uniform_tok_per_s_from_duration():
    row = build_result_row(
        engine="m3",
        concurrency=2,
        max_new_tokens=128,
        token_ids=[[1, 2, 3, 4, 5], [1, 2, 3]],  # decode tokens: 4 + 2 = 6
        engine_decode_tok_per_s=999.0,  # engine's own figure, kept separately
        ttft_ms=None,
        total_global_steps=5,
        resource_report=_report(duration_s=3.0, avg_power_w=None),
    )
    assert row["total_tokens"] == 6
    assert row["uniform_tok_per_s"] == pytest.approx(2.0)
    assert row["engine_decode_tok_per_s"] == 999.0
    assert row["approx_itl_ms"] == pytest.approx(600.0)  # 3.0s / 5 steps * 1000
    assert row["tok_per_s_per_w"] is None  # avg_power_w was None


def test_build_result_row_computes_tok_per_s_per_w_when_power_known():
    row = build_result_row(
        engine="m2",
        concurrency=1,
        max_new_tokens=128,
        token_ids=[[1, 2, 3]],  # 2 decode tokens
        engine_decode_tok_per_s=13.1,
        ttft_ms=523.0,
        total_global_steps=None,
        resource_report=_report(duration_s=1.0, avg_power_w=10.0),
    )
    assert row["uniform_tok_per_s"] == pytest.approx(2.0)
    assert row["tok_per_s_per_w"] == pytest.approx(0.2)
    assert row["approx_itl_ms"] is None  # no total_global_steps for M2
    assert row["ttft_ms"] == 523.0


def test_build_result_row_passes_through_memory_fields():
    row = build_result_row(
        engine="m4",
        concurrency=8,
        max_new_tokens=128,
        token_ids=[[1, 2]],
        engine_decode_tok_per_s=1.0,
        ttft_ms=None,
        total_global_steps=1,
        resource_report=_report(peak_allocated_gb=7.5, peak_reserved_gb=8.0, peak_rss_gb=2.5),
    )
    assert row["peak_allocated_gb"] == 7.5
    assert row["peak_reserved_gb"] == 8.0
    assert row["peak_rss_gb"] == 2.5
