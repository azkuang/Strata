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

Paged KV cache (block-based memory management) lands in M4.

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
**9.448**s.

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
