"""Fast (non-GPU) test for the batched chat-prompt builder: only exercises
the tokenizer, not the model, so it runs in the default (non `gpu`-marked)
test suite.
"""

from transformers import AutoTokenizer

from strata.model import DEFAULT_MODEL_PATH, build_batch_chat_prompt


def test_build_batch_chat_prompt_left_pads_to_longest():
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
    batch = build_batch_chat_prompt(
        tokenizer,
        [
            "Hi.",
            "Write a much longer message here, deliberately padded out with "
            "extra words so it tokenizes to more tokens than the first one.",
        ],
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[0] == 2

    # Left-padding: the shorter sequence's real tokens must be right-aligned,
    # i.e. its attention_mask is a run of 0s followed by a run of 1s.
    row0_mask = attention_mask[0].tolist()
    first_one = row0_mask.index(1)
    assert all(v == 0 for v in row0_mask[:first_one])
    assert all(v == 1 for v in row0_mask[first_one:])

    # The longer prompt should have at least as many real (unpadded) tokens.
    assert attention_mask[1].sum().item() >= attention_mask[0].sum().item()
