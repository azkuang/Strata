#!/usr/bin/env python3
"""M2 demo: run the static-batching engine on multiple prompts at once.

Pass --prompt multiple times to build a batch. Deliberately mix a short and
a long prompt to see the "waste" M2 is meant to make visible: once the
short sequence's answer finishes, the engine keeps running full-batch
forward passes for it (frozen output, pad-token input) until the longest
sequence in the batch finishes too.

Example:
    uv run python scripts/run_m2.py \
        --prompt "Write a one-sentence description of a binary search tree." \
        --prompt "Write a detailed, at-least-five-sentence explanation of how a hash table resolves collisions using open addressing." \
        --max-new-tokens 128 --verbose
"""

import argparse

from strata.engine import BatchEngine
from strata.model import build_batch_chat_prompt, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="User message; repeat --prompt to add more sequences to the batch",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true", help="Print tensor shapes at each step")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    batch = build_batch_chat_prompt(tokenizer, args.prompt, system=args.system)
    engine = BatchEngine(model, tokenizer, verbose=args.verbose)

    result = engine.generate(
        batch["input_ids"], batch["attention_mask"], max_new_tokens=args.max_new_tokens
    )

    for i, text in enumerate(result.texts):
        step = result.finished_at_step[i]
        finished_desc = (
            f"finished at decode step {step}" if step >= 0 else "hit max_new_tokens (no EOS)"
        )
        print(f"\n--- Sequence {i} ({finished_desc}) ---")
        print(text)

    print("\n--- Timing ---")
    print(
        f"TTFT: {result.ttft_ms:.1f} ms | aggregate decode: {result.decode_tok_per_s:.2f} tok/s "
        f"| batch size: {len(result.texts)} | decode steps run: {result.total_decode_steps}"
    )


if __name__ == "__main__":
    main()
