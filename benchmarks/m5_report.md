# M5 — Benchmark & Write-up

**Date:** 2026-07-27
**Raw data:** `benchmarks/raw/m5_results.jsonl` (9 rows, one per grid cell)
**Baseline compared against:** `benchmarks/m0_baseline.md`

## Summary

The from-scratch engine reaches **83.5% of the M0 vLLM bf16 baseline at concurrency 1**
(11.37 vs 13.61 tok/s) and **59.2% at concurrency 32** (251.34 vs 424.23 tok/s), with
M2's static batching the best-performing variant at concurrency 8 and 32 on this workload
(M3 holds a narrow lead at concurrency 1: 11.43 vs M2's 11.37 uniform tok/s, i.e. 84.0% vs
83.5% of the vLLM baseline). The single biggest
takeaway is a result that contradicts the naive M2→M3→M4 ordering: on a *closed* prompt
pool where `max_concurrent_slots == number of requests`, **M3 and M4 are slower than M2**
(198.42 and 160.45 tok/s at concurrency 32), because continuous batching has no queue to
schedule from and therefore collects none of its benefit while still paying 100% of its
per-step KV bookkeeping cost. That bookkeeping cost is directly measurable and scales with
concurrency: per-global-step time from concurrency 1→32 grows +8.7% for M2, +35.3% for M3,
and +63.1% for M4, on a machine where decode is memory-bandwidth-bound. The conclusion is
that the remaining gap to vLLM is not a scheduling problem — it is a *data-movement* problem,
which is precisely what Phase 3's kernel work targets.

## Methodology

- **Model:** Qwen2.5-Coder-7B-Instruct (bf16), same checkpoint as M0–M4. 28 layers,
  4 KV heads (GQA), head_dim 128 → **56 KiB of KV cache per token** across all layers.
- **Workload:** the 8-prompt fixed pool in `strata/benchmark_prompts.py`, cycled to reach
  concurrency 8/32; `--max-new-tokens 128`; greedy (argmax) decoding, no sampling, no
  repetition penalty.
  - **Measured prompt lengths** (after chat templating, tokenized; from an ad-hoc
    verification run, not part of the committed grid data): 41, 40, 43, 51, 50,
    43, 46, 53 tokens → **mean 45.9, min 40, max 53**. At concurrency 8 and 32 the pool
    cycles these same 8 prompts, so M2's left-padded batch is 53 tokens wide in every case.
  - **Measured mean output length per sequence** (derived from each row's `total_tokens`,
    which counts decode-phase tokens only, plus the one token produced by prefill):
    63.0 at concurrency 1 (all engines); 95.75 (M2) / 93.75 (M3, M4) at concurrency 8;
    94.88 (M2) / 92.75 (M3, M4) at concurrency 32. Sequences therefore average roughly
    **46 prompt + 93 generated ≈ 139 tokens** of KV at their longest.
- **Differs from M0's vLLM/llama.cpp workload** (512 synthetic input tokens / 256 output
  tokens via `vllm bench serve`'s random dataset). Our sequences are ~11x shorter on the
  prompt side and ~2.7x shorter on the output side. This comparison is therefore about
  batching-strategy **trends** (throughput scaling, tok/s/W scaling, memory scaling), not
  an exact apples-to-apples absolute-throughput match. Two directional caveats worth
  stating: shorter prompts mean *less* KV to stream per decode step (flattering our
  numbers relative to M0's), while shorter outputs mean prefill is amortized over fewer
  decode steps (penalizing ours). Absolute ratios below should be read as order-of-magnitude
  positioning, not as a precise deficit.
- **Execution:** each (engine, concurrency) cell ran in its own subprocess via
  `scripts/run_m5_suite.py`, so `torch.cuda.max_memory_allocated()` and process RSS reflect
  that config alone and aren't contaminated by a prior config's allocator cache.
  All 9 cells ran with the suite script's defaults — **no deviations, no re-runs, no failed
  cells**. M4 used the default 4000-block pool (`--block-size 16`).
- **No preemption occurred in any M4 grid cell.** M4's `total_global_steps` is identical to
  M3's at every concurrency (63 / 127 / 127), which is the signature of an unconstrained
  pool: a preempted sequence would have to re-prefill and would push M4's step count above
  M3's (as it does in the 14-block pressure scenario discussed under Design Rationale).
  This grid therefore isolates paging's *steady-state bookkeeping* cost with zero recompute
  mixed in.
- **`uniform_tok_per_s`** = total decode tokens / wall-clock duration of the whole
  `generate()` call (prefill/admission included) for every engine — the fair cross-engine
  figure, since M2's own `decode_tok_per_s` excludes its prefill phase while M3/M4's divide
  by their full wall clock. Each engine's self-reported `engine_decode_tok_per_s` is kept
  in the raw JSONL and reproduced below for continuity with M2–M4's own milestone entries.
- **Power** was sampled by polling `nvidia-smi --query-gpu=power.draw` on a background
  thread every 0.5s. Sample counts per cell are small (10–35, recorded as
  `num_power_samples`), so the tok/s/W figures carry meaningful sampling noise — treat
  them as trend indicators, not 3-significant-figure measurements.
- **One measurement artifact, investigated and confirmed benign:** M2 produced slightly
  more total decode tokens than M3/M4 at the same concurrency (758 vs 742 at c=8;
  3004 vs 2936 at c=32). This is *not* a correctness bug. A direct comparison run (an
  ad-hoc verification run, not part of the committed grid data) showed 4 of 8 sequences
  token-for-token identical between M2 and M3, with the other 4 diverging only at decode
  indices 64, 75, 84 and 100 — i.e. deep into generation, where the tiny floating-point
  differences between a padded 8-wide batched matmul and a per-sequence one eventually
  flip an argmax and shift EOS timing by a few tokens. Because `uniform_tok_per_s =
  total_tokens / duration_s` puts the token count in the numerator, this inflates M2's
  `uniform_tok_per_s` by about 2.3% at concurrency 32 relative to M3 at an identical 127
  global steps (3004 vs 2936 tokens: `3004/2936 = 1.0232`); the per-step inter-token-latency
  comparison in the Throughput & Latency table is the step-count-normalized version of this
  comparison and is unaffected by the token-count difference.

## Throughput & Latency

| Engine | Concurrency | uniform tok/s | tok/s (self-reported) | TTFT / approx ITL | wall clock (s) |
|---|---|---|---|---|---|
| M2 (static batching) | 1 | 11.37 | 12.81 | 519.6 ms TTFT / 86.6 ms ITL | 5.455 |
| M2 (static batching) | 8 | 80.42 | 86.40 | 577.1 ms TTFT / 74.2 ms ITL | 9.426 |
| M2 (static batching) | 32 | **251.34** | 273.43 | 891.4 ms TTFT / 94.1 ms ITL | 11.952 |
| M3 (continuous batching) | 1 | 11.43 | 11.44 | — / 86.1 ms ITL | 5.423 |
| M3 (continuous batching) | 8 | 72.51 | 72.52 | — / 80.6 ms ITL | 10.233 |
| M3 (continuous batching) | 32 | 198.42 | 198.45 | — / 116.5 ms ITL | 14.797 |
| M4 (paged KV, 4000 blocks) | 1 | 11.14 | 11.14 | — / 88.3 ms ITL | 5.566 |
| M4 (paged KV, 4000 blocks) | 8 | 66.62 | 66.63 | — / 87.7 ms ITL | 11.138 |
| M4 (paged KV, 4000 blocks) | 32 | 160.45 | 160.48 | — / 144.1 ms ITL | 18.298 |
| vLLM (M0) | 1 | 13.61 | — | 177.6 ms TTFT | — |
| vLLM (M0) | 8 | 129.67 | — | 441.3 ms TTFT | — |
| vLLM (M0) | 32 | 424.23 | — | 859.3 ms TTFT | — |
| llama.cpp (M0) | 1 | 46.63 | — | — | — |
| llama.cpp (M0) | 8 | 295.25 | — | — | — |
| llama.cpp (M0) | 32 | 886.18 | — | — | — |

TTFT is only meaningful for M2, which has one shared prefill phase for the whole batch;
M3/M4 admit sequences at different wall-clock times, so there is no single TTFT and the
approximate ITL (wall clock / global steps, which *includes* amortized admission cost) is
reported instead.

**Ratio to the M0 vLLM bf16 baseline at matched concurrency:**

| Engine | c=1 | c=8 | c=32 |
|---|---|---|---|
| M2 | 83.5% | 62.0% | 59.2% |
| M3 | 84.0% | 55.9% | 46.8% |
| M4 | 81.9% | 51.4% | 37.8% |

(Against llama.cpp's Q4_K_M numbers the same cells land at 24.4% / 27.2% / 28.4% for M2.)

### Discussion

**The predicted M2 < M3 < M4 ordering does not hold, and the reason is a property of the
benchmark, not a defect in M3/M4.** `scripts/benchmark_m5.py` passes
`max_concurrent_slots = concurrency` and supplies exactly `concurrency` prompts. The queue
is therefore empty from global step 0 in every M3/M4 cell: every sequence is admitted in
the initial admission loop and nothing ever waits for a slot. Continuous batching's entire
value proposition — evict a finished sequence and immediately admit a waiting one instead
of letting the batch drain — is unexercised. What remains is only the residual benefit of
shrinking the active batch as sequences finish, which is real but small here (the workload's
finish times are tightly clustered: M2's batch still ran the full 127 global steps at both
c=8 and c=32, and mean output length is ~93 of a 128 budget), against the full cost of
M3's per-step repack. The cost wins.

This is not a hypothetical: M3's own milestone demo, with `max_concurrent_slots=2` and 3
prompts, *did* show the scheduler working (the third prompt admitted at global step 63 as
soon as a slot freed, rather than after the batch drained). The grid in this report simply
does not create that condition at any concurrency level. **A fair statement of the result
is: at offered load ≤ slot count, static batching is the better engine; continuous batching
only pays for itself once offered load exceeds slot count.** That is the honest conclusion
the data supports, and it is a more useful engineering finding than a rigged workload
confirming the expected ordering would have been.

**The gap to vLLM widens with concurrency (83.5% → 59.2% for M2), rather than narrowing.**
At concurrency 1 the from-scratch engine is essentially at parity with vLLM, which is
expected — a single unbatched sequence is bandwidth-bound on streaming the 15 GB of bf16
weights once per token, and there is very little for a serving engine to be clever about.
As concurrency rises, vLLM's advantages (fused/paged attention kernels, CUDA graphs,
chunked prefill, a C++-level scheduler) compound while ours are all in Python. The clearest
single measurement of this is the per-global-step cost:

| Engine | ITL c=1 | ITL c=8 | ITL c=32 | growth c=1→32 |
|---|---|---|---|---|
| M2 | 86.6 ms | 74.2 ms | 94.1 ms | **+8.7%** |
| M3 | 86.1 ms | 80.6 ms | 116.5 ms | **+35.3%** |
| M4 | 88.3 ms | 87.7 ms | 144.1 ms | **+63.1%** |
| vLLM (M0) | 72.8 ms | 59.9 ms | 71.8 ms | −1.4% |

M2's near-flat curve is the physics of the machine showing through: decode is
memory-bandwidth-bound on the weights, so a forward pass over 32 sequences costs almost the
same as over 1 (+8.7%), which is exactly why batching works at all and matches vLLM's own
roughly flat ITL band. **Everything above M2's curve in the M3 and M4 rows is bookkeeping
this engine added, not work the model required.** M3 adds 22.4 ms/step at c=32 and M4 adds
50.0 ms/step — and both scale with concurrency, because both do O(active sequences × layers)
Python-level tensor work per step (`pad_and_batch_caches`/`split_batched_cache` in M3;
`gather_and_batch` in M4, each a nested loop over 28 layers × up to 32 sequences, i.e.
~896 small CUDA ops per step before the model even runs).

## Memory Utilization

| Engine | Concurrency | peak_allocated_gb | peak_reserved_gb | peak_rss_gb |
|---|---|---|---|---|
| M2 | 1 | 15.301 | 15.433 | 5.440 |
| M2 | 8 | 15.446 | 15.680 | 5.440 |
| M2 | 32 | 15.939 | 16.341 | 5.440 |
| M3 | 1 | 15.304 | 15.443 | 5.440 |
| M3 | 8 | 15.424 | 15.636 | 5.440 |
| M3 | 32 | 15.846 | 16.163 | 5.440 |
| M4 | 1 | 18.974 | 19.204 | 5.443 |
| M4 | 8 | 19.094 | 19.394 | 5.440 |
| M4 | 32 | 19.517 | 19.839 | 5.444 |

**Unit caveat:** `peak_allocated_gb` and `peak_reserved_gb` are true **GB** (bytes / 1e9,
from the CUDA allocator), while `peak_rss_gb` is actually **GiB** (`ru_maxrss` KB / 1024²) —
a ~7% unit mismatch inherited from `strata/resource_monitor.py`. The first two columns are
directly comparable to each other; the third is not on the same unit and should not be
differenced against them. It is also a *process*-level figure dominated by the model-loading
path (5.440–5.444 across all nine cells, i.e. carrying no signal about KV behavior), and
on GB10's unified LPDDR5X the CPU and GPU figures are drawing on the same physical ~119 GiB
pool anyway, so they are not additive.

### Discussion

**The prediction that M4's peak allocation would be roughly flat across concurrency while
M2/M3's grew is refuted.** All three engines grow by almost exactly the same absolute
amount from concurrency 1 to 32: **+0.638 GB (M2), +0.542 GB (M3), +0.543 GB (M4)**. M4 is
uniformly ~3.67 GB *above* M3 at every concurrency, but its slope is identical to M3's.

The 3.67 GB offset is fully explained and confirms the pool is doing exactly what it should:
a 4000-block pool at block_size 16, 4 KV heads, head_dim 128, bf16, K and V, 28 layers is
`4000 × 4 × 16 × 128 × 2 bytes × 2 × 28 = 3.670 GB` — matching the observed M4−M3 delta at
concurrency 1 (18.974 − 15.304 = **3.670 GB**) to three decimal places.

The *reason the slope doesn't flatten* is the more interesting finding, and it is the same
scope note the README already flags for M4: **HF's stock `forward()` requires one contiguous
KV tensor per layer, so M4 still materializes a contiguous, left-padded
`[batch, kv_heads, max_len, head_dim]` scratch tensor every step** — the identical transient
allocation M3 builds. M4 has replaced the *persistent* per-sequence caches with a fixed pool,
but it has not removed the *transient* batched gather buffer, and that buffer is what scales
with concurrency. So M4 currently pays the pool's memory cost on top of, not instead of,
the contiguous-batch memory cost. Paged KV's memory win is real in principle (no
padding-to-longest in the persistent store, no reserve-for-worst-case) but cannot be
*collected* until a kernel reads pages in place. That is a direct, measured argument for
Phase 3.

A second observation on sizing: with mean sequence length ~139 tokens (prompt + generated),
32 concurrent sequences need `ceil(139/16) × 32 = 288` of the 4000 blocks — **7.2% pool
utilization**. The "unconstrained" configuration this grid uses is therefore substantially
over-provisioned, leaving roughly 3.41 GB of the 3.67 GB pool idle. In a real deployment the
pool would be sized to the memory actually available after weights, which is exactly the
regime where block pressure and preemption start to matter (see below).

## Efficiency (tokens/sec/watt)

| Engine | Concurrency | avg_power_w | tok/s/W | (power samples) |
|---|---|---|---|---|
| M2 | 1 | 23.819 | 0.477 | 10 |
| M2 | 8 | 25.842 | 3.112 | 18 |
| M2 | 32 | 33.669 | **7.465** | 23 |
| M3 | 1 | 24.325 | 0.470 | 10 |
| M3 | 8 | 25.459 | 2.848 | 19 |
| M3 | 32 | 30.537 | 6.498 | 28 |
| M4 | 1 | 24.438 | 0.456 | 11 |
| M4 | 8 | 24.838 | 2.682 | 21 |
| M4 | 32 | 28.194 | 5.691 | 35 |

**Efficiency scaling, concurrency 1 → 32:**

| Engine | tok/s/W c=1 → c=32 | multiplier | power multiplier | throughput multiplier |
|---|---|---|---|---|
| M2 | 0.477 → 7.465 | **15.6x** | 1.41x | 22.1x |
| M3 | 0.470 → 6.498 | **13.8x** | 1.26x | 17.4x |
| M4 | 0.456 → 5.691 | **12.5x** | 1.15x | 14.4x |
| vLLM (M0) | 0.45 → 9.52 | 21.2x | 1.46x | 31.2x |
| llama.cpp (M0) | — → 19.91 @ c=32 | — | — | — |

### Discussion

**Yes — the from-scratch engine reproduces the batching-efficiency curve, and reproduces
its shape closely.** M2 gains **15.6x** in tokens/sec/watt from batching alone against
vLLM's 21.2x, at a 1.41x power cost against vLLM's 1.46x. The mechanism M0 identified is
confirmed from first principles: batching increases throughput ~22x while power increases
only ~1.4x, because the weight streaming that dominates a decode step is *shared* across
the batch. Reading 15 GB of bf16 weights once to serve 32 sequences instead of 1 is the
entire trick, and it is a bandwidth argument, not a compute argument.

The efficiency multiplier degrades monotonically M2 (15.6x) → M3 (13.8x) → M4 (12.5x), for
the same reason their throughput does: M3/M4's per-step bookkeeping adds work that consumes
wall-clock time. Note the power column tells a subtler story — M3 and M4 draw *less* power
at concurrency 32 (30.5 W and 28.2 W) than M2 (33.7 W), and all three draw well under
vLLM's 44.6 W. That is not a virtue: it means the GPU is idler under our engines, waiting
on Python-side scheduling and small-tensor launches between forward passes. M2 reaches 78%
of vLLM's tok/s/W while reaching only 59% of its throughput precisely *because* our engines
under-drive the hardware. Absolute efficiency, not just the multiplier, is what a real
serving deployment cares about, and there we are still behind.

Set against llama.cpp's **19.91 tok/s/W** at concurrency 32 with Q4_K_M weights, our best
figure of 7.465 is 2.7x off — a larger gap than the 2.1x llama.cpp holds over vLLM. This
lands squarely on M0's central hardware finding: GB10's ~273 GB/s memory bandwidth is ~6.6x
lower than a discrete RTX PRO 6000's, so decode throughput here is set almost entirely by
*how many bytes of weights must cross the memory bus per generated token*. A 4-bit weight
format moves ~4x fewer bytes than bf16 and therefore buys roughly a 2x tok/s/W improvement
essentially for free. **No amount of batching-strategy refinement can recover that factor,
because batching amortizes the weight read across the batch but does not shrink it.** On
this specific machine, the quantization axis and the batching axis are close to orthogonal,
and we have now exhausted most of what the batching axis has to give.

## Design Rationale

The three engines are not a strictly-improving sequence; each trades a different resource,
and the numbers above show which trade the workload actually rewards.

**M2 (static batching) buys throughput with wasted compute and rigid memory.** Batch
membership is fixed for the whole `generate()` call: a sequence that hits EOS is masked
and fed pad tokens rather than evicted, so the batch keeps paying for it until every
sequence finishes. Prompts are left-padded to the longest in the batch, so memory scales
with `batch_size × longest_prompt` rather than with actual tokens. On *this* workload both
costs are small — the 8 prompts span only 40–53 tokens (13 tokens of padding waste at
worst) and mean output length is ~93 of a 128 budget, so finish times cluster tightly and
there is little dead batch time to reclaim. In exchange, M2 gets the cheapest possible
per-step path: HF's `DynamicCache` appends the new token in place, one shared batched
prefill covers all 32 sequences, and per-step cost stays nearly flat (+8.7% from c=1 to
c=32). That flatness is why M2 wins here.

**M3 (continuous batching) buys latency and utilization under *queueing* with per-step
data movement.** Because batch membership changes every step, no single contiguous KV
tensor can persist; each sequence keeps its own unpadded cache and every step must
`pad_and_batch_caches` them into one tensor and `split_batched_cache` them back out.
That is a full read-and-rewrite of the entire active KV working set every single step —
at 56 KiB/token and ~4,400 active tokens at concurrency 32 (32 sequences × ~139 tokens
mean), roughly 250 MB copied out and back per step, on a 273 GB/s bus, plus ~1,800 small
tensor operations of Python/launch overhead. Measured cost: **+22.4 ms per step at
concurrency 32** over M2, i.e. M3 is 21.1% slower than M2 at c=32, 9.8% slower at c=8, and
marginally *ahead* at c=1 (+0.6%, where there is nothing to repack and no batched prefill
for M2 to win with). Admission is also less efficient: M3 prefills each sequence with its own
batch-1 `forward()` call, so 32 sequences means 32 under-utilized prefill launches versus
M2's one. **None of this is bought back on this grid, because the queue is never non-empty.**
The design is correct; the workload simply doesn't ask it the question it answers. Under
offered load exceeding slot count — the condition every real serving system operates in —
M2 would leave a finished sequence's slot idle for the remainder of the batch while M3
refills it immediately, and the ordering would invert.

**M4 (paged KV) buys memory flexibility with indirection.** Paging's promise is eliminating
two forms of waste at once: padding-to-longest in the persistent store, and
reserve-for-worst-case allocation. It delivers on both — a sequence's KV lives in
`ceil(len/16)` non-contiguous physical blocks with no padding, allocated on demand. And
its per-step *write* path is strictly better than M3's: `scatter_new_token` writes only the
one newly-computed token into the pool, where M3 rewrites every token of every sequence
every step. But its per-step *read* path is worse, and the read path dominates:
`gather_and_batch` must walk each sequence's block table, index-select its blocks out of the
pool, reshape, left-pad, and concatenate — a nested loop over 28 layers × up to 32 sequences,
producing the same contiguous scratch tensor M3 produced but reaching it through an extra
level of indirection. Net measured overhead over M3, with **identical global-step counts
(63 / 127 / 127) confirming zero preemption and therefore zero recompute**: **2.6% at
concurrency 1, 8.1% at concurrency 8, 19.1% at concurrency 32.** That number is the clean
answer to "what does paging cost when it isn't buying you anything?" — it is pure
bookkeeping, and it grows with the size of the active set because the gather does.

**Separating paging's two distinct costs.** This grid measures only the first; M4's own
milestone run measured the second. Under deliberate block pressure — a 14-block pool
(224 token slots) with 3 prompts and `max_concurrent_slots=3`, so block availability was
the only queuing constraint — M4 measured **8.57 aggregate decode tok/s over 152 global
steps and 25.782 s wall clock**, against M3's **127 global steps** on the same 3 prompts
(README.md, Benchmarks section). Sequence 1 was preempted shortly after admission and
re-admitted at global step 25, and its discarded progress had to be recomputed from prefill.
The two costs decompose as:

- **Bookkeeping (unconstrained pool, this grid):** 19.1% throughput at concurrency 32,
  entirely per-step, with the step count unchanged from M3.
- **Recompute-based preemption (constrained pool, M4's milestone run):** **+25 global
  steps, a 19.7% increase in total steps**, on top of the bookkeeping — work the engine
  performs twice. (The 152-vs-127 step ratio is the cleanly attributable part; the wall-clock
  gap in that run is larger still, but it was measured at a different concurrency and prompt
  count than any cell here, so it is not directly differenceable against this grid.)

The design choice preemption represents is worth stating plainly, because it is the
interesting one. When the pool cannot give a fresh block to every active sequence that needs
one in the same step, an engine can either (a) reserve each sequence's full worst-case token
budget at admission, or (b) admit optimistically and evict someone when it runs short.
Option (a) never recomputes anything but reintroduces exactly the reserve-for-worst-case
waste paged attention exists to eliminate — with `max_new_tokens=128` and a mean actual
output of 92.75 tokens, roughly **27% of every reservation would go unused**, and far worse
for a workload with a long tail. Option (b) trades a bounded, *measured* recompute cost
(+19.7% steps in the observed case) for the ability to run a pool sized to real demand
rather than worst-case demand. M4 chose (b), and the fact that this grid's 4000-block pool
sat at ~7.2% utilization while a 14-block pool triggered preemption within 25 steps
shows how narrow the window between "wasteful" and "under pressure" actually is — which is
the whole reason the allocator has to handle the pressure case gracefully rather than
assuming it away.

**Why the numbers still point forward and not backward.** The naive reading of this grid —
"M2 won, revert to static batching" — is wrong. M3's and M4's costs are both *implementation*
costs (Python-level per-step tensor marshalling to satisfy a kernel that demands contiguity),
while their benefits are *architectural* (scheduling under load; memory proportional to
actual demand). The architectural benefits survive; the implementation costs are exactly what
a custom kernel removes.

## Bridge to Phase 3

M4's paged, non-contiguous block reads are precisely where a stock HF attention kernel stops
fitting (README.md's M4 scope note), and this benchmark quantifies the consequence rather
than just asserting it: because `forward()` demands one contiguous KV tensor per layer, M4
gathers its blocks back into a padded scratch buffer every step, paying **19.1% throughput
at concurrency 32** and gaining **zero** of paging's memory benefit (its peak allocation
grows +0.543 GB from c=1 to c=32, statistically the same slope as M3's +0.542 GB, while
sitting 3.67 GB higher for the pool). A Triton/CUDA attention kernel that reads pages
directly through the block table deletes both at once — the gather disappears, and the
transient contiguous buffer with it — which is the single highest-value next step for this
engine and the natural Phase 3 opening. But it should not be the *only* one. M0's finding
was that llama.cpp's Q4_K_M weights beat vLLM's bf16 by 2.1–3.4x at matched concurrency and
roughly 2x on tok/s/W (19.91 vs 9.52 at concurrency 32) on this specific hardware, for the
specific reason that GB10 decode is memory-bandwidth-bound at ~273 GB/s and a 4-bit format
simply moves ~4x fewer weight bytes per generated token. This report's efficiency section
confirms the mechanism from the other direction: batching bought M2 a 15.6x tok/s/W
improvement precisely by *amortizing* the weight read, and having taken that amortization
about as far as it goes, the only remaining lever on bytes-per-token is to make the weights
themselves smaller. **A quantized matmul kernel (targeting Blackwell's FP4 path) is therefore
the higher-leverage Phase 3 target than any further batching-only optimization** — and a
paged attention kernel and a quantized weight kernel attack the two halves of the same
bandwidth budget, KV traffic and weight traffic, which is why Phase 3 should pursue both
rather than choosing between them.
