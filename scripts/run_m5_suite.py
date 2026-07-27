#!/usr/bin/env python3
"""M5 benchmark suite: runs scripts/benchmark_m5.py once per (engine,
concurrency) grid cell -- {m2, m3, m4} x {1, 8, 32}, matching M0's
concurrency levels (benchmarks/m0_baseline.md) -- each in its own
subprocess so CUDA peak-memory stats and process RSS aren't contaminated
by a prior config's model load / allocator cache. Appends every row to
one JSONL file.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ENGINES = ["m2", "m3", "m4"]
CONCURRENCIES = [1, 8, 32]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/raw/m5_results.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4000,
        help="m4 pool size; generous enough to avoid preemption at concurrency<=32 (see plan's Global Constraints sizing note)",
    )
    args = parser.parse_args()

    if Path(args.output).exists():
        print(
            f"error: --output path already exists: {args.output}\n"
            "scripts/benchmark_m5.py appends to this file, so re-running the "
            "suite against an existing file would interleave a second run's "
            "rows with the first. Delete/rename the existing file or pass a "
            "different --output path.",
            file=sys.stderr,
        )
        sys.exit(1)

    for engine in ENGINES:
        for concurrency in CONCURRENCIES:
            print(f"\n=== engine={engine} concurrency={concurrency} ===", flush=True)
            cmd = [
                sys.executable,
                "scripts/benchmark_m5.py",
                "--engine", engine,
                "--concurrency", str(concurrency),
                "--max-new-tokens", str(args.max_new_tokens),
                "--output", args.output,
            ]
            if engine == "m4":
                cmd += ["--num-blocks", str(args.num_blocks)]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
