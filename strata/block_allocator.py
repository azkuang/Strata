"""Fixed-size block allocator + KV pool for paged KV cache (M4).

Owns a free list of physical block indices and the KV pool itself: one
preallocated tensor per layer, per key/value, shape
[num_blocks, kv_heads, block_size, head_dim]. alloc()/free() manage the
free list only -- reading/writing pool contents is
strata/paged_kv_cache.py's job (SequenceBlockTable tracks which physical
blocks belong to which sequence and in what order).
"""

import torch


class BlockAllocator:
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: list[int] = list(range(num_blocks))
        self.keys = [
            torch.zeros(num_blocks, kv_heads, block_size, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.values = [
            torch.zeros(num_blocks, kv_heads, block_size, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def num_free(self) -> int:
        return len(self.free_blocks)

    def alloc(self, n: int) -> list[int] | None:
        """Pop `n` free block indices, or return None if fewer than `n`
        are free. All-or-nothing: never allocates a partial set.
        """
        if n == 0:
            return []
        if n > len(self.free_blocks):
            return None
        allocated = self.free_blocks[-n:]
        del self.free_blocks[-n:]
        return allocated

    def free(self, block_ids: list[int]) -> None:
        self.free_blocks.extend(block_ids)
