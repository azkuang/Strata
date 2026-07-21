#!/usr/bin/env python3
"""M1 demo: run the naive single-request engine on a prompt.

Example:
    uv run python scripts/run_m1.py \
        --prompt "Write a C function that reverses a string in place." \
        --max-new-tokens 128 --verbose
"""

import argparse

from strata.engine import NaiveEngine
from strata.model import build_chat_prompt, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="User message to send to the model")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true", help="Print tensor shapes at each step")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    input_ids = build_chat_prompt(tokenizer, args.prompt, system=args.system)
    engine = NaiveEngine(model, tokenizer, verbose=args.verbose)

    result = engine.generate(input_ids, max_new_tokens=args.max_new_tokens)

    print("\n--- Generated text ---")
    print(result.text)
    print("\n--- Timing ---")
    print(f"TTFT: {result.ttft_ms:.1f} ms | decode: {result.decode_tok_per_s:.2f} tok/s "
          f"| tokens generated: {len(result.token_ids)}")


if __name__ == "__main__":
    main()
