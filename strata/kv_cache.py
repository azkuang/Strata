"""Per-sequence KV cache pad/split repacking for ContinuousBatchEngine (M3).

Each active sequence keeps its own unpadded per-layer KV cache tensors
(shape [1, kv_heads, cur_len, head_dim]), since sequences enter/leave the
batch at different iterations and end up at different lengths. HF's
past_key_values is one batched Cache object, so every decode step the
currently-active set must be left-padded into a shared batch, run through
one forward(), then split back into unpadded per-sequence caches for the
next step (whose active set may differ again). This repack cost is exactly
what M4's paged KV cache removes.
"""

import torch
from transformers.cache_utils import DynamicCache

SequenceCache = list[tuple[torch.Tensor, torch.Tensor]]


def pad_and_batch_caches(
    caches: list[SequenceCache],
) -> tuple[DynamicCache, torch.Tensor, torch.Tensor]:
    """Left-pad a list of per-sequence KV caches to a common length and batch them.

    Returns (batched_cache, attention_mask, position_ids):
      - batched_cache: DynamicCache, each layer's keys/values shaped
        [B, kv_heads, max_len, head_dim], left-padded with zeros.
      - attention_mask: [B, max_len] long tensor, 0 for pad, 1 for real.
      - position_ids: [B, 1], the next-token position for each sequence
        (equal to that sequence's current real length).
    """
    batch_size = len(caches)
    num_layers = len(caches[0])
    lengths = [caches[i][0][0].shape[-2] for i in range(batch_size)]
    max_len = max(lengths)
    device = caches[0][0][0].device

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, length in enumerate(lengths):
        attention_mask[i, max_len - length :] = 1

    ddp_cache_data = []
    for layer in range(num_layers):
        keys_list = []
        values_list = []
        for i, length in enumerate(lengths):
            k, v = caches[i][layer]
            pad_len = max_len - length
            if pad_len > 0:
                kv_heads, head_dim = k.shape[1], k.shape[3]
                pad_shape = (1, kv_heads, pad_len, head_dim)
                k = torch.cat([torch.zeros(pad_shape, dtype=k.dtype, device=k.device), k], dim=-2)
                v = torch.cat([torch.zeros(pad_shape, dtype=v.dtype, device=v.device), v], dim=-2)
            keys_list.append(k)
            values_list.append(v)
        ddp_cache_data.append((torch.cat(keys_list, dim=0), torch.cat(values_list, dim=0)))

    batched_cache = DynamicCache(ddp_cache_data=ddp_cache_data)
    position_ids = torch.tensor(lengths, dtype=torch.long, device=device).unsqueeze(-1)
    return batched_cache, attention_mask, position_ids


def split_batched_cache(
    cache: DynamicCache, attention_mask: torch.Tensor
) -> list[SequenceCache]:
    """Inverse of pad_and_batch_caches: split a batched cache back into
    per-sequence unpadded caches. attention_mask must be the mask used for
    the forward() call that produced `cache` (so its row sums give each
    sequence's true post-step length); left-padding means each row's real
    content is right-aligned, so slicing the last `length` columns recovers
    it exactly.
    """
    batch_size = attention_mask.shape[0]
    lengths = attention_mask.sum(dim=-1).tolist()
    num_layers = len(cache.layers)

    result: list[SequenceCache] = [[] for _ in range(batch_size)]
    for layer_idx in range(num_layers):
        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values
        for i in range(batch_size):
            length = int(lengths[i])
            result[i].append((keys[i : i + 1, :, -length:, :], values[i : i + 1, :, -length:, :]))
    return result
