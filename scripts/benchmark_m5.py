#!/usr/bin/env python3
"""M5 benchmark harness: run one (engine, concurrency) configuration end
to end and append one JSONL result row to --output.

scripts/run_m5_suite.py (Task 5) invokes this once per grid cell across
engines {m2, m3, m4} x concurrency {1, 8, 32} -- each in its own
subprocess so CUDA peak-memory stats aren't contaminated across configs.
Run directly for a single ad-hoc measurement, e.g.:

    uv run python scripts/benchmark_m5.py --engine m3 --concurrency 8 \
        --output /tmp/m5_smoke.jsonl
"""

import argparse
import json

from strata.benchmark_prompts import make_prompt_pool
from strata.benchmark_utils import build_result_row
from strata.engine import BatchEngine, ContinuousBatchEngine, PagedContinuousBatchEngine
from strata.model import build_batch_chat_prompt, build_chat_prompt, load_model_and_tokenizer
from strata.resource_monitor import ResourceMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["m2", "m3", "m4"], required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--num-blocks", type=int, default=4000, help="m4 only: KV block pool size"
    )
    parser.add_argument("--block-size", type=int, default=16, help="m4 only: tokens per block")
    parser.add_argument("--output", required=True, help="JSONL file to append the result row to")
    args = parser.parse_args()

    print(f"Loading model... (engine={args.engine}, concurrency={args.concurrency})")
    model, tokenizer = load_model_and_tokenizer()
    prompts = make_prompt_pool(args.concurrency)

    monitor = ResourceMonitor()

    if args.engine == "m2":
        batch = build_batch_chat_prompt(tokenizer, prompts)
        engine = BatchEngine(model, tokenizer)
        with monitor:
            result = engine.generate(
                batch["input_ids"], batch["attention_mask"], max_new_tokens=args.max_new_tokens
            )
        ttft_ms = result.ttft_ms
        total_global_steps = result.total_decode_steps
    else:
        prompt_input_ids = [build_chat_prompt(tokenizer, p) for p in prompts]
        if args.engine == "m3":
            engine = ContinuousBatchEngine(model, tokenizer)
        else:
            engine = PagedContinuousBatchEngine(
                model, tokenizer, num_blocks=args.num_blocks, block_size=args.block_size
            )
        with monitor:
            result = engine.generate(
                prompt_input_ids,
                max_concurrent_slots=args.concurrency,
                max_new_tokens=args.max_new_tokens,
            )
        ttft_ms = None
        total_global_steps = result.total_global_steps

    row = build_result_row(
        engine=args.engine,
        concurrency=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        token_ids=result.token_ids,
        engine_decode_tok_per_s=result.decode_tok_per_s,
        ttft_ms=ttft_ms,
        total_global_steps=total_global_steps,
        resource_report=monitor.report(),
    )

    with open(args.output, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
