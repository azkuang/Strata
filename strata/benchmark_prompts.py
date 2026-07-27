"""Fixed prompt pool for M5 benchmark runs (M2/M3/M4 at concurrency
1/8/32 -- scripts/benchmark_m5.py), so every engine runs the identical
workload at a given concurrency level. Domain-flavored (systems/firmware,
matching this project's specialization angle -- see handoff.md) short-
and long-form prompts, cycled to reach any requested count deterministi-
cally (no randomness), so a rerun is reproducible.
"""

BASE_PROMPTS: list[str] = [
    "Write a one-sentence description of a binary search tree.",
    "Write a one-sentence description of a linked list.",
    "Explain what the MESI cache coherence protocol is in two sentences.",
    "Write a detailed, at-least-five-sentence explanation of how a hash "
    "table resolves collisions using open addressing.",
    "Explain the difference between a UEFI NVRAM variable and a CMOS "
    "setting in three sentences.",
    "Write a short C function that computes the factorial of an integer "
    "recursively.",
    "Explain what a page fault handler does in an operating system, in "
    "three sentences.",
    "Write a detailed, at-least-five-sentence explanation of how "
    "continuous batching improves LLM serving throughput over static "
    "batching.",
]


def make_prompt_pool(n: int) -> list[str]:
    """Return exactly n prompts, cycling through BASE_PROMPTS in order."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return [BASE_PROMPTS[i % len(BASE_PROMPTS)] for i in range(n)]
