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

Block manager and scheduler land in M3/M4 as batching is introduced.

## Benchmarks

M0 baseline (vLLM + llama.cpp reference numbers) recorded in `benchmarks/m0_baseline.md`.
M1's naive engine measured **13.10 tok/s** decode at concurrency 1 (single request),
matching the M0 vLLM concurrency-1 baseline of 13.61 tok/s — expected, since a single
unbatched request is the one case where the naive loop isn't leaving batching
throughput on the table. Full benchmark comparisons resume at M5 once continuous
batching (M3) and paged KV (M4) are in place.

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
