"""Fast (non-GPU, no model load) tests for the per-sequence KV cache
pad/split repacking used by ContinuousBatchEngine (M3): every decode step,
the currently-active sequences' individually-shaped caches must be
left-padded into one batched forward() call, then split back out.
"""

import torch

from strata.kv_cache import pad_and_batch_caches, split_batched_cache


def _make_seq_cache(num_layers: int, kv_heads: int, seq_len: int, head_dim: int):
    return [
        (
            torch.randn(1, kv_heads, seq_len, head_dim),
            torch.randn(1, kv_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    ]


def test_pad_and_batch_caches_left_pads_and_masks():
    seq_short = _make_seq_cache(num_layers=2, kv_heads=2, seq_len=3, head_dim=4)
    seq_long = _make_seq_cache(num_layers=2, kv_heads=2, seq_len=5, head_dim=4)

    batched_cache, attention_mask, position_ids = pad_and_batch_caches([seq_short, seq_long])

    assert attention_mask.shape == (2, 5)
    assert attention_mask[0].tolist() == [0, 0, 1, 1, 1]
    assert attention_mask[1].tolist() == [1, 1, 1, 1, 1]
    assert position_ids.squeeze(-1).tolist() == [3, 5]
    assert batched_cache.layers[0].keys.shape == (2, 2, 5, 4)
    assert batched_cache.layers[1].keys.shape == (2, 2, 5, 4)


def test_split_batched_cache_round_trips_immediately_after_padding():
    seq_short = _make_seq_cache(num_layers=2, kv_heads=2, seq_len=3, head_dim=4)
    seq_long = _make_seq_cache(num_layers=2, kv_heads=2, seq_len=5, head_dim=4)

    batched_cache, attention_mask, _ = pad_and_batch_caches([seq_short, seq_long])
    split = split_batched_cache(batched_cache, attention_mask)

    for layer_idx in range(2):
        assert torch.equal(split[0][layer_idx][0], seq_short[layer_idx][0])
        assert torch.equal(split[0][layer_idx][1], seq_short[layer_idx][1])
        assert torch.equal(split[1][layer_idx][0], seq_long[layer_idx][0])
        assert torch.equal(split[1][layer_idx][1], seq_long[layer_idx][1])


def test_pad_and_batch_caches_single_sequence_no_padding():
    seq = _make_seq_cache(num_layers=1, kv_heads=2, seq_len=4, head_dim=4)
    batched_cache, attention_mask, position_ids = pad_and_batch_caches([seq])

    assert attention_mask.tolist() == [[1, 1, 1, 1]]
    assert position_ids.squeeze(-1).tolist() == [4]
    assert torch.equal(batched_cache.layers[0].keys, seq[0][0])
