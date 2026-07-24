"""Per-sequence block-table bookkeeping and gather/scatter for the paged
KV cache (M4). Physical KV storage lives in BlockAllocator's pool
(strata/block_allocator.py); this module maps between a sequence's logical
token positions and the pool's physical (block, offset) slots, and
reconstructs/updates pool contents around each forward() call.

Every decode step: gather_and_batch() reads each active sequence's tokens
out of the pool via its block table, left-pads/batches across the active
set into a contiguous DynamicCache (same shape contract as M3's
strata.kv_cache.pad_and_batch_caches, so forward() itself doesn't change)
-- then, after forward() runs, scatter_new_token() writes each sequence's
newly-computed token straight into its blocks. Unlike M3, old tokens are
never rewritten: gather only reads, and only the new token is written
each step.
"""

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from strata.block_allocator import BlockAllocator


@dataclass
class SequenceBlockTable:
    block_ids: list[int] = field(default_factory=list)
    length: int = 0


def alloc_blocks_for_length(allocator: BlockAllocator, length: int) -> list[int] | None:
    """Allocate the blocks needed to hold `length` tokens
    (ceil(length / block_size)), or None if not enough free blocks are
    currently available. All-or-nothing, via BlockAllocator.alloc.
    """
    num_blocks_needed = (length + allocator.block_size - 1) // allocator.block_size
    return allocator.alloc(num_blocks_needed)


def write_prefill_tokens(
    allocator: BlockAllocator,
    table: SequenceBlockTable,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> None:
    """Write a freshly-prefilled sequence's full KV into table.block_ids,
    which must already hold enough blocks for keys[0].shape[-2] tokens
    (see alloc_blocks_for_length). Assumes table.length == 0 (a prefill
    always starts a fresh table).

    keys/values: per-layer list, each shaped [1, kv_heads, prompt_len, head_dim].
    """
    block_size = allocator.block_size
    prompt_len = keys[0].shape[-2]
    num_layers = len(allocator.keys)
    num_blocks_used = (prompt_len + block_size - 1) // block_size
    device = allocator.keys[0].device
    block_id_tensor = torch.tensor(table.block_ids[:num_blocks_used], dtype=torch.long, device=device)

    for layer in range(num_layers):
        k = keys[layer][0]  # [kv_heads, prompt_len, head_dim]
        v = values[layer][0]
        kv_heads, _, head_dim = k.shape
        pad_len = num_blocks_used * block_size - prompt_len
        if pad_len > 0:
            pad_shape = (kv_heads, pad_len, head_dim)
            k = torch.cat([k, torch.zeros(pad_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(pad_shape, dtype=v.dtype, device=v.device)], dim=1)
        k = k.reshape(kv_heads, num_blocks_used, block_size, head_dim).permute(1, 0, 2, 3)
        v = v.reshape(kv_heads, num_blocks_used, block_size, head_dim).permute(1, 0, 2, 3)
        allocator.keys[layer][block_id_tensor] = k
        allocator.values[layer][block_id_tensor] = v
    table.length += prompt_len


def gather_and_batch(
    allocator: BlockAllocator, tables: list[SequenceBlockTable]
) -> tuple[DynamicCache, torch.Tensor, torch.Tensor]:
    """Reconstruct each sequence's tokens from its blocks, then left-pad and
    batch them into one DynamicCache for a shared forward() call.

    Returns (batched_cache, attention_mask, position_ids), identical in
    shape/meaning to strata.kv_cache.pad_and_batch_caches's return value.
    """
    batch_size = len(tables)
    num_layers = len(allocator.keys)
    block_size = allocator.block_size
    lengths = [t.length for t in tables]
    max_len = max(lengths)
    device = allocator.keys[0].device

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, length in enumerate(lengths):
        attention_mask[i, max_len - length :] = 1

    ddp_cache_data = []
    for layer in range(num_layers):
        keys_list = []
        values_list = []
        for table in tables:
            length = table.length
            num_blocks_used = (length + block_size - 1) // block_size
            block_id_tensor = torch.tensor(
                table.block_ids[:num_blocks_used], dtype=torch.long, device=device
            )
            gathered_k = allocator.keys[layer][block_id_tensor]  # [nb, kv_heads, block_size, head_dim]
            gathered_v = allocator.values[layer][block_id_tensor]
            kv_heads = gathered_k.shape[1]
            head_dim = gathered_k.shape[3]
            k = gathered_k.permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)[:, :length, :]
            v = gathered_v.permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)[:, :length, :]
            pad_len = max_len - length
            if pad_len > 0:
                pad_shape = (kv_heads, pad_len, head_dim)
                k = torch.cat([torch.zeros(pad_shape, dtype=k.dtype, device=device), k], dim=1)
                v = torch.cat([torch.zeros(pad_shape, dtype=v.dtype, device=device), v], dim=1)
            keys_list.append(k.unsqueeze(0))
            values_list.append(v.unsqueeze(0))
        ddp_cache_data.append((torch.cat(keys_list, dim=0), torch.cat(values_list, dim=0)))

    batched_cache = DynamicCache(ddp_cache_data=ddp_cache_data)
    position_ids = torch.tensor(lengths, dtype=torch.long, device=device).unsqueeze(-1)
    return batched_cache, attention_mask, position_ids


def scatter_new_token(
    allocator: BlockAllocator,
    tables: list[SequenceBlockTable],
    new_keys: list[torch.Tensor],
    new_values: list[torch.Tensor],
) -> None:
    """Write each sequence's newly-computed token K/V into its blocks,
    allocating a fresh block first if this token starts one. Raises
    RuntimeError if a sequence needs a new block but none are free --
    callers (PagedContinuousBatchEngine) must have already confirmed
    enough free blocks exist for every sequence about to take a decode
    step before calling this.

    new_keys/new_values: per-layer list, each shaped
    [batch, kv_heads, 1, head_dim] -- one new token's K/V per sequence
    (same order as `tables`), for every layer, e.g.
    out.past_key_values.layers[l].keys[:, :, -1:, :].
    """
    block_size = allocator.block_size
    num_layers = len(allocator.keys)

    for i, table in enumerate(tables):
        position = table.length
        block_offset = position % block_size
        if block_offset == 0:
            new_block = allocator.alloc(1)
            if new_block is None:
                raise RuntimeError("no free blocks left to extend sequence")
            table.block_ids.append(new_block[0])
        block_id = table.block_ids[position // block_size]

        for layer in range(num_layers):
            allocator.keys[layer][block_id, :, block_offset, :] = new_keys[layer][i, :, 0, :]
            allocator.values[layer][block_id, :, block_offset, :] = new_values[layer][i, :, 0, :]

        table.length += 1
