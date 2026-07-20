# Firmware-Specialized LLM Serving Engine

A from-scratch LLM inference engine, specialized for firmware/UEFI code, with custom low-power inference kernels.

## Why this project

Most local-LLM projects call an existing serving stack (vLLM, Ollama, llama.cpp) and stop there. This one builds the serving engine itself — continuous batching, paged KV-cache memory management, custom GPU kernels — to demonstrate systems engineering applied to ML infrastructure, not just ML usage. It's paired with a firmware-code specialization (fine-tuning on EDK2/UEFI source) as a domain angle that's uncommon in general ML portfolios.

The paged KV-cache work in particular is a direct application of OS virtual-memory/demand-paging concepts to attention memory management — same mental model, different address space.

## Architecture

*To be filled in as Phase 1 progresses — engine design, block manager, scheduler.*

## Benchmarks

*To be filled in starting at M0. Will track tokens/sec, time-to-first-token, inter-token latency, and memory utilization vs. vLLM/llama.cpp baselines, plus throughput vs. GPU power cap.*

## Getting started

```bash
# TODO: environment setup instructions once M0 is complete
```

## References

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM paper)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [EDK2 / TianoCore](https://github.com/tianocore/edk2)

## License

TBD
