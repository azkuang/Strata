"""M1/M2: naive single-request engine and static-batching engine.

Manual prefill + decode loop with a manually-threaded KV cache (transformers'
DynamicCache — one growing [batch, kv_heads, seq, head_dim] tensor per layer).

NaiveEngine (M1) handles a single sequence at a time, built for correctness
and to see the tensor shapes at each step, not for performance. BatchEngine
(M2) batches N sequences with left-padding and a fixed-size static batch —
no scheduler, no mid-batch eviction (see BatchEngine's docstring).
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


@dataclass
class BatchGenerationResult:
    token_ids: list[list[int]]
    texts: list[str]
    ttft_ms: float
    decode_tok_per_s: float
    finished_at_step: list[int]
    total_decode_steps: int
    prefill_shapes: dict = field(default_factory=dict)


class BatchEngine:
    """Static-batch greedy decoding engine.

    Batches N sequences with left-padding to the longest prompt and runs
    one shared forward() per step for the whole batch. Batch size and
    padded slot count are fixed for the life of a generate() call: once a
    sequence hits EOS its output is frozen and it's fed a pad token going
    forward, rather than being evicted from the batch. This wastes
    compute/memory on short sequences once others in the batch are still
    running — intentional, see handoff.md M2 (motivates M3's continuous
    batching).
    """

    def __init__(self, model, tokenizer, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose

    @torch.inference_mode()
    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 256
    ) -> BatchGenerationResult:
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_size = input_ids.shape[0]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Standard HF left-padding recipe: position of each real token is
        # its running count of real tokens so far, minus 1; padded slots get
        # a dummy position (masked out of attention anyway by attention_mask).
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # --- Prefill ---
        t0 = time.perf_counter()
        out = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=True
        )
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        next_ids = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
        ttft_ms = (time.perf_counter() - t0) * 1000

        prefill_shapes = {
            "input_ids": tuple(input_ids.shape),
            "logits": tuple(out.logits.shape),
            "kv_cache_per_layer": tuple(cache.layers[0].keys.shape),
        }
        if self.verbose:
            print(f"[prefill] input_ids={prefill_shapes['input_ids']} "
                  f"logits={prefill_shapes['logits']} "
                  f"kv_cache[0]={prefill_shapes['kv_cache_per_layer']}")

        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished_at_step = [-1] * batch_size
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        is_eos = next_ids.squeeze(-1) == eos_token_id
        for i in range(batch_size):
            if is_eos[i]:
                finished_at_step[i] = 0
            else:
                generated[i].append(next_ids[i, 0].item())
        finished = finished | is_eos

        cur_attention_mask = attention_mask
        next_position = position_ids[:, -1:] + 1

        # --- Decode loop: whole batch advances together every step ---
        t_decode_start = time.perf_counter()
        total_decode_steps = 0
        for step in range(1, max_new_tokens):
            if finished.all():
                break
            total_decode_steps += 1

            # Finished sequences get fed pad_token_id instead of their
            # (stale/eos) id — the value doesn't affect other sequences
            # (no cross-batch attention) and their output is already frozen.
            step_input = torch.where(finished.unsqueeze(-1), pad_token_id, next_ids)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones(batch_size, 1, dtype=cur_attention_mask.dtype, device=device)],
                dim=-1,
            )

            out = self.model(
                step_input,
                attention_mask=cur_attention_mask,
                position_ids=next_position,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            next_ids = logits.argmax(dim=-1, keepdim=True)
            next_position = next_position + 1

            if step == 1 and self.verbose:
                print(f"[decode step 0] input_ids={tuple(step_input.shape)} "
                      f"logits={tuple(out.logits.shape)} "
                      f"kv_cache[0]={tuple(cache.layers[0].keys.shape)}")

            newly_finished = (next_ids.squeeze(-1) == eos_token_id) & (~finished)
            for i in range(batch_size):
                if newly_finished[i]:
                    finished_at_step[i] = step
                elif not finished[i]:
                    generated[i].append(next_ids[i, 0].item())
            finished = finished | newly_finished

        decode_elapsed = time.perf_counter() - t_decode_start
        total_decode_tokens = sum(max(len(g) - 1, 0) for g in generated)
        decode_tok_per_s = (
            total_decode_tokens / decode_elapsed if decode_elapsed > 0 and total_decode_tokens > 0 else 0.0
        )

        texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]
        return BatchGenerationResult(
            token_ids=generated,
            texts=texts,
            ttft_ms=ttft_ms,
            decode_tok_per_s=decode_tok_per_s,
            finished_at_step=finished_at_step,
            total_decode_steps=total_decode_steps,
            prefill_shapes=prefill_shapes,
        )
