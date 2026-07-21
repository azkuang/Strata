"""M1 correctness gate: NaiveEngine's manual prefill+decode loop must produce
token-for-token identical output to HF's greedy `.generate()`, since greedy
decoding is deterministic. Any mismatch means the manual KV-cache threading
is wrong.
"""

import pytest
import torch

from strata.engine import NaiveEngine
from strata.model import build_chat_prompt, load_model_and_tokenizer

MAX_NEW_TOKENS = 32


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_model_and_tokenizer()


@pytest.mark.gpu
def test_naive_engine_matches_hf_greedy_generate(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    input_ids = build_chat_prompt(
        tokenizer, "Write a one-sentence description of a binary search tree."
    )

    engine = NaiveEngine(model, tokenizer)
    result = engine.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)

    with torch.inference_mode():
        hf_out = model.generate(
            input_ids.to(next(model.parameters()).device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    hf_new_tokens = hf_out[0, input_ids.shape[1]:].tolist()
    # Strip any trailing eos HF appends that our loop stops before emitting.
    if tokenizer.eos_token_id in hf_new_tokens:
        hf_new_tokens = hf_new_tokens[: hf_new_tokens.index(tokenizer.eos_token_id)]

    assert result.token_ids == hf_new_tokens
