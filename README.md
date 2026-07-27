# Firmware-Specialized LLM Serving Engine

A from-scratch LLM inference engine, specialized for firmware/UEFI code, with custom low-power inference kernels.

## Why this project

Most local-LLM projects call an existing serving stack (vLLM, Ollama, llama.cpp) and stop there. This one builds the serving engine itself — continuous batching, paged KV-cache memory management, custom GPU kernels — to demonstrate systems engineering applied to ML infrastructure, not just ML usage. It's paired with a firmware-code specialization (fine-tuning on EDK2/UEFI source) as a domain angle that's uncommon in general ML portfolios.

The paged KV-cache work in particular is a direct application of OS virtual-memory/demand-paging concepts to attention memory management — same mental model, different address space.

## Architecture

**M1 (naive single-request engine)** is implemented in `strata/`:

- `strata/model.py` — loads the model/tokenizer (`AutoModelForCausalLM` + `.to("cuda")`,
  no `.generate()`) and builds chat-formatted prompts via the tokenizer's chat template.
- `strata/engine.py` — `NaiveEngine`: a manual prefill + greedy decode loop. Prefill
  calls `model.forward()` once over the full prompt and gets back a `DynamicCache`
  (one growing `[batch, kv_heads, seq, head_dim]` tensor per layer); each decode step
  feeds the single last token back in along with that cache and threads the updated
  cache to the next step. No batching, no scheduler — one sequence at a time, by
  design, to nail down correctness and tensor shapes before M2/M3 add batching.
- `scripts/run_m1.py` — CLI demo; prints generated text plus TTFT/decode tok-per-s,
  and (with `--verbose`) the tensor shapes at prefill and the first decode step.
- `tests/test_m1_correctness.py` — the correctness gate: asserts `NaiveEngine`'s greedy
  output is token-for-token identical to HF's `model.generate(do_sample=False)`, since
  greedy decoding is deterministic and any mismatch means the manual KV-cache threading
  is wrong.

**M2 (static batching)** adds to `strata/`:

- `strata/model.py` — `build_batch_chat_prompt()`: left-pads a batch of chat-templated
  prompts to the longest one (`tokenizer.padding_side = "left"`), so every sequence's
  next-token position lands at the same trailing index across the batch.
- `strata/engine.py` — `BatchEngine`: prefill and decode run as one shared `forward()`
  call per step for the whole batch, using the standard HF left-padding recipe for
  `attention_mask`/`position_ids`. Batch size is fixed for the call: a sequence that
  hits EOS has its output frozen and is fed a pad token for the rest of the batch's
  decode loop rather than being evicted — wasted compute on short sequences once
  others are still running, which M3's continuous batching removes.
- `scripts/run_m2.py` — CLI demo; pass `--prompt` multiple times to build a batch,
  prints per-sequence output and which decode step each sequence finished at.
- `tests/test_batch_prompt.py` / `tests/test_m2_correctness.py` — padding-shape check
  and the correctness gate (token-for-token match against HF's batched
  `model.generate()`), same pattern as M1's test.

**M3 (continuous batching)** adds to `strata/`:

- `strata/kv_cache.py` — `pad_and_batch_caches()` / `split_batched_cache()`: since
  batch membership changes every decode step (sequences finish and get evicted, new
  ones get admitted from a queue), each active sequence keeps its own unpadded
  per-layer KV cache; every step, the active set is left-padded into one shared
  batch for `forward()`, then split back into unpadded per-sequence caches for the
  next step, whose active set may differ again.
- `strata/engine.py` — `ContinuousBatchEngine`: evicts a sequence the instant it
  finishes (EOS or its own `max_new_tokens` budget) and admits the next queued
  prompt into the freed slot via its own single-sequence prefill call, rather than
  waiting for the whole batch to drain like M2. `max_concurrent_slots` caps how many
  sequences run at once; the rest wait in a queue.
- `scripts/run_m3.py` — CLI demo; pass `--prompt` multiple times and
  `--max-concurrent-slots` to force queueing, prints per-sequence output plus which
  global step each sequence was admitted/finished at.
- `tests/test_kv_cache_repack.py` — fast, non-GPU tests for the pad/split repacking
  using synthetic tensors. `tests/test_m3_correctness.py` — the correctness gate:
  each sequence's output must match its own standalone HF `.generate()`, and asserts
  the scheduler actually queued something (not a degenerate static batch).

**M4 (paged KV cache)** adds to `strata/`:

- `strata/block_allocator.py` — `BlockAllocator`: a free list of physical
  block indices plus the KV pool itself (one preallocated
  `[num_blocks, kv_heads, block_size, head_dim]` tensor per layer, per K/V).
  `alloc()`/`free()` manage the free list; exhaustion returns `None`
  rather than raising, since running out of blocks is normal backpressure.
- `strata/paged_kv_cache.py` — `SequenceBlockTable` (a sequence's ordered
  list of physical block indices) plus `gather_and_batch()` /
  `scatter_new_token()`: gather reconstructs each active sequence's tokens
  from its blocks and left-pads/batches across the active set (same shape
  contract as M3's `pad_and_batch_caches`, so `forward()` itself doesn't
  change); scatter writes only the newly-computed token into the pool —
  unlike M3, old tokens are never rewritten every step.
- `strata/engine.py` — `PagedContinuousBatchEngine`: the same admit/evict/
  queue scheduling as `ContinuousBatchEngine` (M3), but admission
  allocates `ceil(prompt_len / block_size)` blocks up front, and a
  sequence can be held in the queue by block-pool pressure, not just by
  `max_concurrent_slots`. If the pool can't cover every already-active
  sequence that needs a fresh block in the same step, one is preempted
  (recompute-based: its progress is discarded and it re-prefills later)
  rather than letting everyone wait, which could deadlock.
- `scripts/run_m4.py` — CLI demo; adds `--num-blocks`/`--block-size` on
  top of M3's args, sized to force visible block-pressure queuing.
- `tests/test_block_allocator.py` / `tests/test_paged_kv_cache.py` — fast,
  non-GPU tests for the allocator's free-list bookkeeping and the
  gather/scatter round trip. `tests/test_m4_correctness.py` — the
  correctness gate: each sequence's output must match its own standalone
  HF `.generate()`, with the block pool deliberately sized so at least one
  sequence is held back by block pressure specifically (not just the
  slot-limit queuing M3's test already covers).

**Scope note:** HF's stock `forward()` needs each layer's KV cache as one
contiguous tensor, so M4 still gathers blocks into a contiguous scratch
tensor before every `forward()` call — it builds the memory-management
half of paged attention (block allocator, block table, free list), not a
custom kernel that reads pages directly. That's Phase 3's job.

## Benchmarks

M0 baseline (vLLM + llama.cpp reference numbers) recorded in `benchmarks/m0_baseline.md`.
M1's naive engine measured **13.10 tok/s** decode at concurrency 1 (single request),
matching the M0 vLLM concurrency-1 baseline of 13.61 tok/s — expected, since a single
unbatched request is the one case where the naive loop isn't leaving batching
throughput on the table. Full benchmark comparisons resume at M5 once continuous
batching (M3) and paged KV (M4) are in place.

M2's static-batching engine measured **22.96** aggregate decode tok/s on a 2-prompt
batch (short + long prompt mixed deliberately), TTFT **545.9** ms; sequence 0 finished
at decode step 63 while the batch kept running 127 steps total for the still-active
sequence — the wasted compute M3 (continuous batching) is designed to remove.

M3's continuous-batching engine measured **23.39** aggregate decode tok/s across
the same 2 prompts as M2 plus a third, with `max_concurrent_slots=2` forcing the
third prompt to queue — it was admitted at global step **63**, as soon
as a slot freed, rather than waiting for the whole batch to drain. Total wall clock:
**9.448**s. Note this figure isn't strictly like-for-like with M2's: M3's wall clock
includes each sequence's individual prefill/admission time, while M2's aggregate
tok/s is computed over its decode-only window — a byproduct of continuous batching
having no single decode-only phase once admissions are interleaved throughout.

M4's paged-KV-cache engine measured **8.57** aggregate decode tok/s on the
same 3 prompts as M3, with a block pool sized to `14` blocks (block_size
16) and `max_concurrent_slots=3` (i.e. no slot-count limit) — small enough
that sequence 1 was held back by block pressure alone, preempted shortly
after its initial admission and re-admitted at global step **25** once
enough blocks freed up. Total wall clock: **25.782**s over 152 global
steps (vs. 127 for the same prompts under M3, since the preempted
sequence's discarded progress must be recomputed from prefill once
re-admitted — the throughput cost of choosing recompute-based preemption
over reserving each sequence's full worst-case budget upfront, which would
avoid this cost but reintroduce the reserve-for-worst-case memory waste
paged attention exists to eliminate). All 3 sequences' final generated
text is identical to M3's output regardless of preemption, confirming
paging/scheduling changes don't affect correctness — only which physical
blocks and how much recomputation a sequence's tokens pass through on the
way there.

**M5** ran the full M2/M3/M4 x concurrency-{1,8,32} grid
(`scripts/run_m5_suite.py`) against the same M0 baseline. Full
throughput/latency/memory/efficiency comparison and design rationale:
`benchmarks/m5_report.md`; raw rows in `benchmarks/raw/m5_results.jsonl`.
Headline: **M2's static batching reached 251.34 uniform tok/s at
concurrency 32 — 59.2% of the M0 vLLM bf16 baseline's 424.23 tok/s — and
beat both M3 (198.42) and M4 (160.45) on this workload.** That inversion
is the report's main finding, and it's a property of the benchmark rather
than a defect in M3/M4: the harness sets `max_concurrent_slots` equal to
the request count, so the queue is never non-empty and continuous
batching has nothing to schedule, while still paying its full per-step KV
repack cost. The cost is directly visible in per-global-step time from
concurrency 1 to 32: **+8.7% for M2, +35.3% for M3, +63.1% for M4** —
M2's near-flat curve is the machine's bandwidth-bound decode showing
through, and everything above it is bookkeeping the engine added.
M2 also reproduced the M0 batching-efficiency curve from first
principles: **15.6x tokens/sec/watt** from concurrency 1 to 32 at a 1.41x
power cost, against vLLM's 21.2x at 1.46x. Two predictions were refuted
by the data and are documented as such: the expected M2 < M3 < M4
throughput ordering did not hold, for the harness reason above, and M4's
peak allocation does *not* flatten across concurrency (it grows +0.543 GB
from c=1 to c=32, the same slope as M3's +0.542 GB, while sitting exactly
3.67 GB higher for the 4000-block pool), because HF's contiguous-`forward()`
requirement means M4 still materializes M3's padded scratch tensor every
step — paging's memory win can't be collected until a kernel reads pages
in place, which is Phase 3's job.

## Getting started

```bash
uv sync

# Run the M1 naive engine on a prompt
uv run python scripts/run_m1.py --prompt "Write a C function that reverses a string in place." --verbose

# Run the correctness test suite
uv run pytest tests/ -v
```

## References

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM paper)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [EDK2 / TianoCore](https://github.com/tianocore/edk2)

## License

TBD
