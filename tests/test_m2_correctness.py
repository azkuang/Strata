"""M2 correctness gate: BatchEngine's manual batched prefill+decode loop
must produce token-for-token identical output, per sequence, to HF's
batched greedy `.generate()` with left-padding. Uses two prompts of very
different lengths so the batch actually exercises padding and (likely)
different per-sequence finish steps.
"""

import pytest
import torch

from strata.engine import BatchEngine
from strata.model import build_batch_chat_prompt, load_model_and_tokenizer

MAX_NEW_TOKENS = 32


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_model_and_tokenizer()


@pytest.mark.gpu
def test_batch_engine_matches_hf_batched_greedy_generate(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompts = [
        "Write a one-sentence description of a binary search tree.",
        "Write a detailed, at-least-five-sentence explanation of how a hash "
        "table resolves collisions using open addressing.",
    ]
    batch = build_batch_chat_prompt(tokenizer, prompts)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    engine = BatchEngine(model, tokenizer)
    result = engine.generate(input_ids, attention_mask, max_new_tokens=MAX_NEW_TOKENS)

    device = next(model.parameters()).device
    with torch.inference_mode():
        hf_out = model.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    prompt_len = input_ids.shape[1]
    for i, our_tokens in enumerate(result.token_ids):
        hf_new_tokens = hf_out[i, prompt_len:].tolist()
        # HF's generate() forces pad_token_id (or eos_token_id) into the
        # sequence once it finishes; strip trailing eos/pad the same way
        # our engine stops emitting before them.
        for stop_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            if stop_id in hf_new_tokens:
                hf_new_tokens = hf_new_tokens[: hf_new_tokens.index(stop_id)]
        assert our_tokens == hf_new_tokens, f"sequence {i} mismatch"
