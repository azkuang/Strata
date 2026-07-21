"""M1: naive single-request engine.

Manual prefill + decode loop with a manually-threaded KV cache (transformers'
DynamicCache — one growing [batch, kv_heads, seq, head_dim] tensor per layer).
No batching, no scheduler: this is a single sequence at a time, built for
correctness and to see the tensor shapes at each step, not for performance.
"""

import time
from dataclasses import dataclass, field

import torch


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    ttft_ms: float
    decode_tok_per_s: float
    prefill_shapes: dict = field(default_factory=dict)
    first_decode_shapes: dict = field(default_factory=dict)


class NaiveEngine:
    """Single-request greedy decoding engine, one forward() call at a time."""

    def __init__(self, model, tokenizer, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose

    def _cache_layer_shapes(self, cache) -> list[tuple[int, ...]]:
        return [tuple(layer.keys.shape) for layer in cache.layers]

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> GenerationResult:
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        eos_token_id = self.tokenizer.eos_token_id

        # --- Prefill ---
        t0 = time.perf_counter()
        out = self.model(input_ids, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
        ttft_ms = (time.perf_counter() - t0) * 1000

        prefill_shapes = {
            "input_ids": tuple(input_ids.shape),
            "logits": tuple(out.logits.shape),
            "kv_cache_per_layer": self._cache_layer_shapes(cache),
        }
        if self.verbose:
            print(f"[prefill] input_ids={prefill_shapes['input_ids']} "
                  f"logits={prefill_shapes['logits']} "
                  f"kv_cache[0]={prefill_shapes['kv_cache_per_layer'][0]}")

        generated: list[int] = []
        first_decode_shapes: dict = {}
        if next_id.item() != eos_token_id:
            generated.append(next_id.item())

        # --- Decode loop ---
        t_decode_start = time.perf_counter()
        for step in range(max_new_tokens - 1):
            if generated and generated[-1] == eos_token_id:
                break

            out = self.model(next_id, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)

            if step == 0:
                first_decode_shapes = {
                    "input_ids": tuple(next_id.shape),
                    "logits": tuple(out.logits.shape),
                    "kv_cache_per_layer": self._cache_layer_shapes(cache),
                }
                if self.verbose:
                    print(f"[decode step 0] input_ids={first_decode_shapes['input_ids']} "
                          f"logits={first_decode_shapes['logits']} "
                          f"kv_cache[0]={first_decode_shapes['kv_cache_per_layer'][0]}")

            tok = next_id.item()
            if tok == eos_token_id:
                break
            generated.append(tok)

        decode_elapsed = time.perf_counter() - t_decode_start
        decode_tok_per_s = (len(generated) - 1) / decode_elapsed if decode_elapsed > 0 and len(generated) > 1 else 0.0

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            token_ids=generated,
            text=text,
            ttft_ms=ttft_ms,
            decode_tok_per_s=decode_tok_per_s,
            prefill_shapes=prefill_shapes,
            first_decode_shapes=first_decode_shapes,
        )
