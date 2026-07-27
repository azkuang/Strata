"""Fast (non-GPU) tests for strata.resource_monitor's power-sampling loop
and duration tracking, using an injected fake power_sample_fn so these
don't depend on nvidia-smi or a real GPU. A separate @pytest.mark.gpu test
covers real torch.cuda peak-memory integration.
"""

import time

import pytest
import torch

from strata.resource_monitor import ResourceMonitor, current_peak_rss_gb


def test_power_sampling_collects_and_averages_injected_samples():
    def fake_power():
        return 10.0

    monitor = ResourceMonitor(power_sample_fn=fake_power, interval_s=0.01)
    with monitor:
        time.sleep(0.05)
    report = monitor.report()

    assert report.num_power_samples >= 2
    assert report.avg_power_w == pytest.approx(10.0)
    assert report.duration_s >= 0.05


def test_power_sampling_ignores_none_samples():
    calls = iter([10.0, None, 20.0, None, 30.0])

    def fake_power():
        return next(calls, None)

    monitor = ResourceMonitor(power_sample_fn=fake_power, interval_s=0.01)
    with monitor:
        time.sleep(0.05)
    report = monitor.report()

    assert report.num_power_samples <= 3
    for value in (report.avg_power_w,):
        assert value is None or 10.0 <= value <= 30.0


def test_report_without_entering_context_has_zero_duration_and_no_samples():
    monitor = ResourceMonitor(power_sample_fn=lambda: 5.0)
    report = monitor.report()
    assert report.duration_s == 0.0
    assert report.num_power_samples == 0
    assert report.avg_power_w is None


def test_peak_memory_is_none_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monitor = ResourceMonitor(power_sample_fn=lambda: None, interval_s=0.01)
    with monitor:
        pass
    report = monitor.report()
    assert report.peak_allocated_gb is None
    assert report.peak_reserved_gb is None


def test_current_peak_rss_gb_is_positive():
    assert current_peak_rss_gb() > 0.0


@pytest.mark.gpu
def test_peak_allocated_memory_reflects_a_real_cuda_allocation():
    monitor = ResourceMonitor(power_sample_fn=lambda: None, interval_s=0.05)
    with monitor:
        tensor = torch.zeros(64 * 1024 * 1024, dtype=torch.float32, device="cuda")  # ~256MB
        tensor += 1  # force materialization
    report = monitor.report()
    assert report.peak_allocated_gb is not None
    assert report.peak_allocated_gb > 0.2
