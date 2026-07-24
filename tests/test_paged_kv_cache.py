"""Fast (non-GPU) tests for paged KV cache gather/scatter (M4): synthetic
tensors and a CPU-backed BlockAllocator. Mirrors
tests/test_kv_cache_repack.py's style for M3's pad/split repacking --
this is the M4 equivalent, minus the padding cost every step since old
tokens are never rewritten, only gathered.
"""

import pytest
import torch

from strata.block_allocator import BlockAllocator
from strata.paged_kv_cache import (
    SequenceBlockTable,
    alloc_blocks_for_length,
    gather_and_batch,
    scatter_new_token,
    write_prefill_tokens,
)

NUM_LAYERS = 2
KV_HEADS = 2
HEAD_DIM = 4
BLOCK_SIZE = 4


def _make_allocator(num_blocks: int) -> BlockAllocator:
    return BlockAllocator(
        num_blocks=num_blocks,
        num_layers=NUM_LAYERS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=torch.float32,
        device="cpu",
    )


def _make_prompt_kv(prompt_len: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    keys = [torch.randn(1, KV_HEADS, prompt_len, HEAD_DIM) for _ in range(NUM_LAYERS)]
    values = [torch.randn(1, KV_HEADS, prompt_len, HEAD_DIM) for _ in range(NUM_LAYERS)]
    return keys, values


def test_write_prefill_then_gather_round_trips_a_single_sequence():
    allocator = _make_allocator(num_blocks=4)
    prompt_len = 5  # spans 2 blocks at block_size=4 (4 + 1)
    keys, values = _make_prompt_kv(prompt_len)

    table = SequenceBlockTable(block_ids=alloc_blocks_for_length(allocator, prompt_len))
    write_prefill_tokens(allocator, table, keys, values)
    assert table.length == prompt_len
    assert len(table.block_ids) == 2

    batched_cache, attention_mask, position_ids = gather_and_batch(allocator, [table])
    assert attention_mask.tolist() == [[1] * prompt_len]
    assert position_ids.squeeze(-1).tolist() == [prompt_len]
    for layer in range(NUM_LAYERS):
        assert torch.equal(batched_cache.layers[layer].keys[0], keys[layer][0])
        assert torch.equal(batched_cache.layers[layer].values[0], values[layer][0])


def test_gather_and_batch_left_pads_across_sequences_of_different_lengths():
    allocator = _make_allocator(num_blocks=4)
    keys_short, values_short = _make_prompt_kv(3)
    keys_long, values_long = _make_prompt_kv(5)

    table_short = SequenceBlockTable(block_ids=alloc_blocks_for_length(allocator, 3))
    write_prefill_tokens(allocator, table_short, keys_short, values_short)
    table_long = SequenceBlockTable(block_ids=alloc_blocks_for_length(allocator, 5))
    write_prefill_tokens(allocator, table_long, keys_long, values_long)

    _, attention_mask, position_ids = gather_and_batch(allocator, [table_short, table_long])
    assert attention_mask.shape == (2, 5)
    assert attention_mask[0].tolist() == [0, 0, 1, 1, 1]
    assert attention_mask[1].tolist() == [1, 1, 1, 1, 1]
    assert position_ids.squeeze(-1).tolist() == [3, 5]


def test_scatter_new_token_allocates_a_fresh_block_on_boundary_and_writes_correctly():
    allocator = _make_allocator(num_blocks=4)
    prompt_len = BLOCK_SIZE  # exactly fills one block -- next token must allocate a new one
    keys, values = _make_prompt_kv(prompt_len)
    table = SequenceBlockTable(block_ids=alloc_blocks_for_length(allocator, prompt_len))
    write_prefill_tokens(allocator, table, keys, values)
    assert len(table.block_ids) == 1
    assert allocator.num_free() == 3

    new_keys = [torch.randn(1, KV_HEADS, 1, HEAD_DIM) for _ in range(NUM_LAYERS)]
    new_values = [torch.randn(1, KV_HEADS, 1, HEAD_DIM) for _ in range(NUM_LAYERS)]
    scatter_new_token(allocator, [table], new_keys, new_values)

    assert len(table.block_ids) == 2  # rolled over into a new block
    assert table.length == prompt_len + 1
    assert allocator.num_free() == 2

    batched_cache, _, _ = gather_and_batch(allocator, [table])
    for layer in range(NUM_LAYERS):
        expected_k = torch.cat([keys[layer][0], new_keys[layer][0]], dim=1)
        expected_v = torch.cat([values[layer][0], new_values[layer][0]], dim=1)
        assert torch.equal(batched_cache.layers[layer].keys[0], expected_k)
        assert torch.equal(batched_cache.layers[layer].values[0], expected_v)


def test_scatter_new_token_within_block_does_not_allocate():
    allocator = _make_allocator(num_blocks=4)
    prompt_len = 2  # leaves room in the same block (block_size=4)
    keys, values = _make_prompt_kv(prompt_len)
    table = SequenceBlockTable(block_ids=alloc_blocks_for_length(allocator, prompt_len))
    write_prefill_tokens(allocator, table, keys, values)
    free_before = allocator.num_free()

    new_keys = [torch.randn(1, KV_HEADS, 1, HEAD_DIM) for _ in range(NUM_LAYERS)]
    new_values = [torch.randn(1, KV_HEADS, 1, HEAD_DIM) for _ in range(NUM_LAYERS)]
    scatter_new_token(allocator, [table], new_keys, new_values)

    assert len(table.block_ids) == 1
    assert allocator.num_free() == free_before


def test_scatter_new_token_raises_when_pool_exhausted():
    allocator = _make_allocator(num_blocks=1)
    prompt_len = BLOCK_SIZE
    keys, values = _make_prompt_kv(prompt_len)
    table = SequenceBlockTable(block_ids=alloc_blocks_for_length(allocator, prompt_len))
    write_prefill_tokens(allocator, table, keys, values)
    assert allocator.num_free() == 0

    new_keys = [torch.randn(1, KV_HEADS, 1, HEAD_DIM) for _ in range(NUM_LAYERS)]
    new_values = [torch.randn(1, KV_HEADS, 1, HEAD_DIM) for _ in range(NUM_LAYERS)]

    with pytest.raises(RuntimeError):
        scatter_new_token(allocator, [table], new_keys, new_values)
