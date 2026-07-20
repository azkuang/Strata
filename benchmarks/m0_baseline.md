# M0 Baseline — Environment & Reference Benchmarks

**Date:** 2026-07-20
**Model:** Qwen/Qwen2.5-Coder-7B-Instruct (bf16 for vLLM; official Q4_K_M GGUF for llama.cpp)

This is the baseline that milestones M1–M5 (the from-scratch engine) are compared against.
Every number below is measured on the actual dev machine, not vendor spec.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GB10 (Grace-Blackwell superchip), compute capability sm_121 |
| CPU | 20-core Grace (Cortex-X925), aarch64, Ubuntu 24.04.4 |
| Memory | ~119 GiB unified LPDDR5X, coherent CPU/GPU |
| CUDA / driver | CUDA 13.0 / driver 580.142 |
| Power control | None — `nvidia-smi -pl` / `-lgc` both N/A. `power.draw` readable and used below. |


## Software stack

- PyTorch 2.11.0+cu130 (aarch64, pinned by vLLM's dependency resolution)
- vLLM 0.25.1 (installed via `uv add vllm==0.25.1` with `index-strategy = "unsafe-best-match"`
  and `tool.uv.environments` restricted to `linux`+`aarch64` — see `pyproject.toml`)
- llama.cpp built from source (commit `91d2fc3`) with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`
  (cmake auto-resolved this to `121a` for GB10)

**Setup gotcha:** vLLM's engine failed to start with `fatal error: Python.h: No such file or
directory` during a JIT-compiled CUDA util step. Fixed by installing the `python3-dev` system
package (not a Python dependency — a system header package, so it needed `sudo apt install`).

## vLLM baseline (bf16, `vllm bench serve`, chat completions)

Server: `vllm serve models/Qwen2.5-Coder-7B-Instruct --served-model-name qwen2.5-coder-7b-instruct`
Workload: 512 input tokens / 256 output tokens per request, `random` dataset.

| Concurrency | Output tok/s | Mean TTFT (ms) | Mean ITL (ms) | Avg GPU power (W) | tok/s/W |
|---|---|---|---|---|---|
| 1  | 13.61  | 177.6 | 72.8 | 30.6 | 0.45 |
| 8  | 129.67 | 441.3 | 59.9 | 32.6 | 3.98 |
| 32 | 424.23 | 859.3 | 71.8 | 44.6 | 9.52 |

Raw results: `benchmarks/raw/m0_concurrency{1,8,32}.json`, power trace `benchmarks/raw/power_samples.csv`.

**Observations:**
- Throughput scales ~31x from concurrency 1→32 while power only scales ~1.5x (30.6W → 44.6W) —
  tokens/sec/watt improves ~21x. This is the batching-efficiency curve M2/M3 are meant to
  reproduce from first principles.
- ITL is roughly flat (~60–73ms) across concurrency levels — decode is memory-bandwidth-bound,
  not compute-bound, so adding concurrent sequences doesn't much change per-token latency until
  the GPU is saturated.

## llama.cpp baseline (Q4_K_M GGUF, `llama-batched-bench`)

Same workload shape (512 prompt / 256 generated tokens), using `-npl` to vary parallel sequences.

| Parallel sequences | Gen tok/s (S_TG) | Avg GPU power (W) | tok/s/W |
|---|---|---|---|
| 1  | 46.63  | — (not sampled) | — |
| 8  | 295.25 | — (not sampled) | — |
| 32 | 886.18 | 44.3 | 19.91 |

Raw results: `benchmarks/raw/llama_batched_bench.log`, power trace `benchmarks/raw/llama_power_samples.csv`
(power only sampled for the npl=32 run).

**Observation:** the 4-bit-quantized llama.cpp baseline is ~2.1–3.4x faster than the bf16 vLLM
baseline at matched concurrency (e.g. 46.63 vs 13.61 tok/s at concurrency 1; 886 vs 424 tok/s at
32) at similar power draw (44.3W vs 44.6W at concurrency 32), i.e. roughly 2x better tok/s/W.
This is expected and informative for this hardware specifically: GB10 decode is memory-bandwidth-
bound (~273 GB/s spec, ~6.6x lower than a discrete RTX PRO 6000), so a 4-bit weight format moves
proportionally less data per generated token. This is a concrete argument for why quantization-
aware design matters more here than it would on a bandwidth-rich discrete GPU — worth carrying
into M4/M5 and the Phase 3 kernel work.

## What M1–M5 are compared against

- **Throughput target:** close the gap to 424 tok/s (vLLM, concurrency 32) with a from-scratch
  continuous-batching + paged-KV engine; 886 tok/s (llama.cpp, quantized) is the stretch/quantized
  comparison point once quantization is in scope.
- **Latency target:** ITL in the 60–75ms band at moderate-to-high concurrency.
- **Efficiency target:** tok/s/W should follow the same batching-driven curve seen above (0.45 →
  3.98 → 9.52 for vLLM) — M5 should reproduce and ideally improve on this curve, not just match
  raw throughput.
