"""Background power/memory sampling for M5 benchmark runs.

Wraps a benchmark's engine.generate() call to record average GPU power
draw (via nvidia-smi, sampled on a background thread), peak CUDA
allocator memory, and peak process RSS -- the three signals handoff.md's
M5 scope calls for, given nvidia-smi reports unified-memory total as N/A
on GB10 (see benchmarks/m0_baseline.md's Hardware section).
"""

from __future__ import annotations

import resource
import subprocess
import threading
import time
from dataclasses import dataclass

import torch


def read_nvidia_smi_power_draw_w() -> float | None:
    """One instantaneous GPU power.draw reading in watts, or None if
    nvidia-smi isn't available/parseable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        return float(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def current_peak_rss_gb() -> float:
    """Peak resident set size of this process since it started, in GB.

    ru_maxrss is monotonically non-decreasing for the life of the process
    (Linux reports it in KB), so calling this once at the end of a
    benchmark run -- one engine/concurrency config per process, per
    scripts/benchmark_m5.py -- gives that run's peak RSS without needing
    a background sampler.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


@dataclass
class ResourceReport:
    duration_s: float
    avg_power_w: float | None
    num_power_samples: int
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None
    peak_rss_gb: float


class ResourceMonitor:
    """Context manager: samples GPU power on a background thread for the
    duration of the `with` block, and reads peak CUDA allocator stats +
    peak process RSS at/after exit via .report().

    `power_sample_fn` is injectable so tests can exercise the sampling
    loop/averaging without a real GPU or nvidia-smi; defaults to
    read_nvidia_smi_power_draw_w.
    """

    def __init__(self, power_sample_fn=None, interval_s: float = 0.5):
        self._power_sample_fn = power_sample_fn or read_nvidia_smi_power_draw_w
        self._interval_s = interval_s
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._duration_s = 0.0

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            watts = self._power_sample_fn()
            if watts is not None:
                self._samples.append(watts)
            self._stop_event.wait(self._interval_s)

    def __enter__(self) -> "ResourceMonitor":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 4)
        if self._start_time is not None:
            self._duration_s = time.monotonic() - self._start_time

    def report(self) -> ResourceReport:
        peak_allocated_gb = None
        peak_reserved_gb = None
        if torch.cuda.is_available():
            peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        return ResourceReport(
            duration_s=self._duration_s,
            avg_power_w=(sum(self._samples) / len(self._samples)) if self._samples else None,
            num_power_samples=len(self._samples),
            peak_allocated_gb=peak_allocated_gb,
            peak_reserved_gb=peak_reserved_gb,
            peak_rss_gb=current_peak_rss_gb(),
        )
