#!/usr/bin/env python3
"""M4 demo: run the paged-KV-cache continuous-batching engine on multiple
prompts with a capped block pool, so admission can be held back by block
pressure (not just max_concurrent_slots, M3's queuing mechanism) -- pass
--num-blocks small enough relative to the prompts/--max-new-tokens to see
it happen.

Example (same 3 prompts as M3's demo, with a small pool to force block
pressure via preemption -- max_concurrent_slots=3 admits all 3 immediately,
so any admission after step 0 proves the pool, not slot limits, held a
sequence back):
    uv run python scripts/run_m4.py \
        --prompt "Write a one-sentence description of a binary search tree." \
        --prompt "Write a detailed, at-least-five-sentence explanation of how a hash table resolves collisions using open addressing." \
        --prompt "Write a one-sentence description of a linked list." \
        --max-concurrent-slots 3 --num-blocks 14 --max-new-tokens 128 --verbose
"""

import argparse

from strata.engine import PagedContinuousBatchEngine
from strata.model import build_chat_prompt, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="User message; repeat --prompt to add more sequences to the queue",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-concurrent-slots", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, required=True, help="Total blocks in the KV pool")
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    parser.add_argument("--verbose", action="store_true", help="Print tensor shapes at the first step")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    prompt_input_ids = [build_chat_prompt(tokenizer, p, system=args.system) for p in args.prompt]
    engine = PagedContinuousBatchEngine(
        model, tokenizer, num_blocks=args.num_blocks, block_size=args.block_size, verbose=args.verbose
    )

    result = engine.generate(
        prompt_input_ids,
        max_concurrent_slots=args.max_concurrent_slots,
        max_new_tokens=args.max_new_tokens,
    )

    for i, text in enumerate(result.texts):
        finished = result.finished_at_step[i]
        admitted = result.admitted_at_step[i]
        finished_desc = (
            f"finished at local step {finished}" if finished >= 0 else "hit max_new_tokens (no EOS)"
        )
        print(f"\n--- Sequence {i} (admitted at global step {admitted}, {finished_desc}) ---")
        print(text)

    print("\n--- Timing ---")
    print(
        f"wall clock: {result.wall_clock_s:.3f}s | aggregate decode: {result.decode_tok_per_s:.2f} tok/s "
        f"| sequences: {len(result.texts)} | max_concurrent_slots: {args.max_concurrent_slots} "
        f"| num_blocks: {args.num_blocks} (block_size={args.block_size}) "
        f"| global steps run: {result.total_global_steps}"
    )


if __name__ == "__main__":
    main()
