"""M3 correctness gate: ContinuousBatchEngine's iteration-level scheduling
(evict on finish, admit from a queue into freed slots) must not change any
individual sequence's output vs. running it alone — batching/scheduling is
purely a throughput technique. Uses 3 prompts with max_concurrent_slots=2
so at least one prompt is queued behind the initial admissions, exercising
the actual eviction/admission path (not just a degenerate static batch).

The oracle `model.generate(..., do_sample=False, ...)` calls pass
repetition_penalty=1.0 to override this checkpoint's generation_config.json
default (repetition_penalty=1.1). repetition_penalty is a LogitsProcessor
applied by HF's generate() regardless of do_sample (only temperature/top_p/
top_k are gated on do_sample=True) -- left at its checkpoint default, the
oracle would silently do penalized-greedy decoding, not pure greedy argmax,
which is what ContinuousBatchEngine (manual forward() + argmax, no
processors) actually implements. Confirmed by reproducing the checkpoint
default's oracle output bit-for-bit with a manual repetition-penalty-adjusted
argmax loop outside the engine.
"""

import pytest
import torch

from strata.engine import ContinuousBatchEngine
from strata.model import build_chat_prompt, load_model_and_tokenizer

MAX_NEW_TOKENS = 32

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
def test_continuous_batch_engine_matches_standalone_hf_greedy_generate(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = next(model.parameters()).device
    prompt_input_ids = [build_chat_prompt(tokenizer, p) for p in PROMPTS]

    engine = ContinuousBatchEngine(model, tokenizer)
    result = engine.generate(prompt_input_ids, max_concurrent_slots=2, max_new_tokens=MAX_NEW_TOKENS)

    # Scheduling was actually exercised: at least one sequence was admitted
    # after step 0 (i.e. queued behind the initial fill), not everyone
    # starting immediately (which would degenerate to a static batch).
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
