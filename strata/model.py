"""Model + tokenizer loading for the from-scratch engine."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-Coder-7B-Instruct"


def load_model_and_tokenizer(
    path: str | Path = DEFAULT_MODEL_PATH,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Load the model and tokenizer used by the engine.

    Uses .from_pretrained + .to(device) rather than device_map="auto" since
    this is a single-GPU box and we want a plain nn.Module to call
    forward() on directly (no accelerate hooks in the way).
    """
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model = model.to(device).eval()
    return model, tokenizer


def build_chat_prompt(tokenizer, user_message: str, system: str | None = None) -> torch.Tensor:
    """Apply the tokenizer's chat template and return input_ids of shape [1, S]."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # This transformers version returns a BatchEncoding (dict-like) rather
    # than a bare tensor even with return_dict's default; normalize either way.
    input_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") and not torch.is_tensor(encoded) else encoded
    return input_ids


def build_batch_chat_prompt(
    tokenizer, user_messages: list[str], system: str | None = None
) -> dict:
    """Apply the tokenizer's chat template to a batch of prompts, left-padded
    to the longest one.

    Left-padding (not right-padding) is required for batched causal-LM
    decode: it keeps every sequence's "next token to generate" position at
    the same trailing index across the batch, so a single argmax over the
    batch's last-position logits lines up with the right sequence.

    Builds each prompt's chat-templated text first (tokenize=False), then
    tokenizes the whole batch together with padding=True — two separate
    steps rather than passing a list straight to apply_chat_template, since
    that combined path's kwargs vary across transformers versions (see the
    BatchEncoding gotcha from M1 in handoff.md).
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    texts = []
    for user_message in user_messages:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        texts.append(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        )

    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
