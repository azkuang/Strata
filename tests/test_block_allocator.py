"""Fast (non-GPU) tests for the paged-KV block allocator (M4): free-list
alloc/free bookkeeping and pool tensor shapes only. Content correctness of
what's written into the pool is covered by tests/test_paged_kv_cache.py.
"""

from strata.block_allocator import BlockAllocator


def _make_allocator(num_blocks: int) -> BlockAllocator:
    return BlockAllocator(
        num_blocks=num_blocks, num_layers=2, kv_heads=2, head_dim=4, device="cpu"
    )


def test_alloc_returns_requested_count_and_shrinks_free_list():
    allocator = _make_allocator(num_blocks=4)
    block_ids = allocator.alloc(3)
    assert block_ids is not None
    assert len(block_ids) == 3
    assert allocator.num_free() == 1


def test_alloc_returns_none_when_not_enough_free_blocks():
    allocator = _make_allocator(num_blocks=4)
    assert allocator.alloc(3) is not None
    assert allocator.alloc(2) is None
    assert allocator.num_free() == 1  # failed alloc must not consume blocks


def test_free_returns_blocks_to_the_free_list_for_reuse():
    allocator = _make_allocator(num_blocks=2)
    block_ids = allocator.alloc(2)
    assert allocator.num_free() == 0

    allocator.free(block_ids)
    assert allocator.num_free() == 2

    reallocated = allocator.alloc(2)
    assert reallocated is not None
    assert sorted(reallocated) == sorted(block_ids)


def test_alloc_with_zero_returns_empty_list_without_consuming_blocks():
    allocator = _make_allocator(num_blocks=4)
    initial_free = allocator.num_free()
    result = allocator.alloc(0)
    assert result == []
    assert allocator.num_free() == initial_free


def test_pool_tensors_have_expected_shape():
    allocator = BlockAllocator(
        num_blocks=5, num_layers=3, kv_heads=2, head_dim=4, block_size=16, device="cpu"
    )
    assert len(allocator.keys) == 3
    assert len(allocator.values) == 3
    assert allocator.keys[0].shape == (5, 2, 16, 4)
    assert allocator.values[0].shape == (5, 2, 16, 4)
