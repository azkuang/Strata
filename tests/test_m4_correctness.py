"""M4 correctness gate: PagedContinuousBatchEngine's block-pool-backed
scheduling (including any preemption it triggers) must not change any
individual sequence's final output vs. running it alone -- paging is
purely a memory-management technique. Sizes the block pool so it can hold
any 2 of the 3 prompts' full worst-case budget (prompt + max_new_tokens)
but not all 3 at once, with max_concurrent_slots=3 (i.e. no slot-count
limit) -- prompts are short enough that all 3 likely admit immediately
(slot limits aren't the constraint), so it's block pressure during decode
-- and any preemption/restart it triggers -- that must show up as an
admission after step 0, not the initial admission queueing
test_m3_correctness.py already covers.

Same repetition_penalty=1.0 oracle override as test_m3_correctness.py --
see that file's module docstring for why (this checkpoint's
generation_config.json default of repetition_penalty=1.1 is applied by
HF's generate() regardless of do_sample, which would otherwise make the
oracle do penalized-greedy, not pure-greedy, decoding).
"""

import pytest
import torch

from strata.engine import PagedContinuousBatchEngine
from strata.model import build_chat_prompt, load_model_and_tokenizer

MAX_NEW_TOKENS = 32
BLOCK_SIZE = 16

PROMPTS = [
    "Write a one-sentence description of a binary search tree.",
    "Write a detailed, at-least-five-sentence explanation of how a hash "
    "table resolves collisions using open addressing.",
    "Write a one-sentence description of a linked list.",
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_model_and_tokenizer()


@pytest.mark.gpu
def test_paged_engine_matches_standalone_hf_greedy_generate_under_block_pressure(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = next(model.parameters()).device
    prompt_input_ids = [build_chat_prompt(tokenizer, p) for p in PROMPTS]

    blocks_needed = [
        (ids.shape[-1] + MAX_NEW_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE for ids in prompt_input_ids
    ]
    num_blocks = sum(sorted(blocks_needed)[:2])

    engine = PagedContinuousBatchEngine(
        model, tokenizer, num_blocks=num_blocks, block_size=BLOCK_SIZE
    )
    result = engine.generate(prompt_input_ids, max_concurrent_slots=3, max_new_tokens=MAX_NEW_TOKENS)

    assert any(step > 0 for step in result.admitted_at_step)

    for i, prompt_ids in enumerate(prompt_input_ids):
        with torch.inference_mode():
            hf_out = model.generate(
                prompt_ids.to(device),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.0,
                use_cache=True,
            )
        hf_new_tokens = hf_out[0, prompt_ids.shape[1] :].tolist()
        if tokenizer.eos_token_id in hf_new_tokens:
            hf_new_tokens = hf_new_tokens[: hf_new_tokens.index(tokenizer.eos_token_id)]
        assert result.token_ids[i] == hf_new_tokens, f"sequence {i} mismatch"
