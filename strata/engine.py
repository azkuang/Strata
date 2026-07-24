"""M1/M2/M3/M4: naive single-request, static-batching, continuous-batching,
and paged-KV-cache engines.

Manual prefill + decode loop with a manually-threaded KV cache (transformers'
DynamicCache — one growing [batch, kv_heads, seq, head_dim] tensor per layer).

NaiveEngine (M1) handles a single sequence at a time, built for correctness
and to see the tensor shapes at each step, not for performance. BatchEngine
(M2) batches N sequences with left-padding and a fixed-size static batch —
no scheduler, no mid-batch eviction (see BatchEngine's docstring).
ContinuousBatchEngine (M3) adds iteration-level scheduling: a sequence is
evicted the instant it finishes and the next queued sequence is admitted
into the freed slot, without waiting for the rest of the batch to drain
(see ContinuousBatchEngine's docstring, and strata/kv_cache.py for the
per-step pad/split repacking this requires).

PagedContinuousBatchEngine (M4) replaces ContinuousBatchEngine's per-step
pad/split repack with a fixed-size block pool (BlockAllocator) and
per-sequence block table: gather reconstructs each active sequence's
tokens from its blocks for the shared forward() call, and scatter writes
only the newly-computed token back -- old tokens are never rewritten (see
strata/paged_kv_cache.py and strata/block_allocator.py). When the pool
can't cover every active sequence that needs a fresh block in the same
step, one is preempted (its progress discarded, requeued to re-prefill
later) rather than letting everyone wait, which could deadlock.
"""

import time
from dataclasses import dataclass, field

import torch

from strata.kv_cache import pad_and_batch_caches, split_batched_cache
from strata.block_allocator import BlockAllocator
from strata.paged_kv_cache import (
    SequenceBlockTable,
    alloc_blocks_for_length,
    gather_and_batch,
    scatter_new_token,
    write_prefill_tokens,
)


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


@dataclass
class ContinuousBatchGenerationResult:
    token_ids: list[list[int]]
    texts: list[str]
    finished_at_step: list[int]
    admitted_at_step: list[int]
    total_global_steps: int
    wall_clock_s: float
    decode_tok_per_s: float


@dataclass
class _ActiveSequence:
    seq_id: int
    cache: list[tuple[torch.Tensor, torch.Tensor]]
    next_token: torch.Tensor
    admitted_at_step: int
    local_step: int = 0


class ContinuousBatchEngine:
    """Iteration-level scheduling: evicts a sequence the moment it finishes
    and admits the next queued sequence into the freed slot, without
    waiting for the whole batch to drain (unlike BatchEngine/M2, where
    batch membership is fixed for the whole call). Every decode step,
    the currently-active sequences' individually-shaped KV caches are
    left-padded into one shared forward() call via strata.kv_cache, then
    split back into unpadded per-sequence caches for the next step.
    """

    def __init__(self, model, tokenizer, verbose: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose

    def _prefill(self, prompt_input_ids: torch.Tensor):
        out = self.model(prompt_input_ids, use_cache=True)
        cache = out.past_key_values
        num_layers = len(cache.layers)
        seq_cache = [(cache.layers[l].keys, cache.layers[l].values) for l in range(num_layers)]
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        return seq_cache, next_token

    @torch.inference_mode()
    def generate(
        self,
        prompt_input_ids: list[torch.Tensor],
        max_concurrent_slots: int,
        max_new_tokens: int = 256,
    ) -> ContinuousBatchGenerationResult:
        device = next(self.model.parameters()).device
        eos_token_id = self.tokenizer.eos_token_id
        n = len(prompt_input_ids)

        queue = list(range(n))
        token_ids: list[list[int]] = [[] for _ in range(n)]
        finished_at_step = [-1] * n
        admitted_at_step = [-1] * n
        active: list[_ActiveSequence] = []
        global_step = 0

        def admit(seq_id: int, step: int) -> _ActiveSequence | None:
            ids = prompt_input_ids[seq_id].to(device)
            seq_cache, next_token = self._prefill(ids)
            admitted_at_step[seq_id] = step
            tok = next_token.item()
            if tok == eos_token_id:
                finished_at_step[seq_id] = 0
                return None
            token_ids[seq_id].append(tok)
            return _ActiveSequence(
                seq_id=seq_id, cache=seq_cache, next_token=next_token, admitted_at_step=step
            )

        t0 = time.perf_counter()

        while queue and len(active) < max_concurrent_slots:
            seq = admit(queue.pop(0), global_step)
            if seq is not None:
                active.append(seq)

        while active:
            batch_caches = [seq.cache for seq in active]
            batched_cache, attention_mask, position_ids = pad_and_batch_caches(batch_caches)
            step_attention_mask = torch.cat(
                [attention_mask, torch.ones(len(active), 1, dtype=attention_mask.dtype, device=device)],
                dim=-1,
            )
            step_input = torch.cat([seq.next_token for seq in active], dim=0)

            out = self.model(
                step_input,
                attention_mask=step_attention_mask,
                position_ids=position_ids,
                past_key_values=batched_cache,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            next_ids = logits.argmax(dim=-1, keepdim=True)
            split_caches = split_batched_cache(out.past_key_values, step_attention_mask)

            global_step += 1
            if self.verbose and global_step == 1:
                print(
                    f"[global step {global_step}] active={len(active)} "
                    f"step_input={tuple(step_input.shape)} "
                    f"kv_cache[0]={tuple(batched_cache.layers[0].keys.shape)}"
                )

            still_active = []
            for i, seq in enumerate(active):
                seq.cache = split_caches[i]
                seq.next_token = next_ids[i : i + 1]
                seq.local_step += 1
                tok = next_ids[i, 0].item()
                is_eos = tok == eos_token_id
                # Prefill already contributed 1 token (in admit()), so the
                # decode loop may only contribute max_new_tokens - 1 more to
                # keep the per-sequence total at max_new_tokens (same
                # convention as NaiveEngine's `range(max_new_tokens - 1)`
                # and BatchEngine's `range(1, max_new_tokens)`). local_step
                # reaching max_new_tokens - 1 means this iteration's token
                # is the last one allowed.
                hit_budget = seq.local_step >= max_new_tokens - 1
                # The decode loop always runs at least one iteration once a
                # sequence is active, even when max_new_tokens == 1 (where
                # decode shouldn't contribute any tokens at all). Gate the
                # append on the same budget so that edge case still yields
                # exactly max_new_tokens tokens total instead of max_new_tokens + 1.
                if not is_eos and seq.local_step < max_new_tokens:
                    token_ids[seq.seq_id].append(tok)
                if is_eos or hit_budget:
                    finished_at_step[seq.seq_id] = seq.local_step
                else:
                    still_active.append(seq)
            active = still_active

            while queue and len(active) < max_concurrent_slots:
                seq = admit(queue.pop(0), global_step)
                if seq is not None:
                    active.append(seq)

        wall_clock_s = time.perf_counter() - t0
        total_tokens = sum(max(len(g) - 1, 0) for g in token_ids)
        decode_tok_per_s = (
            total_tokens / wall_clock_s if wall_clock_s > 0 and total_tokens > 0 else 0.0
        )

        texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in token_ids]
        return ContinuousBatchGenerationResult(
            token_ids=token_ids,
            texts=texts,
            finished_at_step=finished_at_step,
            admitted_at_step=admitted_at_step,
            total_global_steps=global_step,
            wall_clock_s=wall_clock_s,
            decode_tok_per_s=decode_tok_per_s,
        )


@dataclass
class _PagedActiveSequence:
    seq_id: int
    table: SequenceBlockTable
    next_token: torch.Tensor
    admitted_at_step: int
    local_step: int = 0


class PagedContinuousBatchEngine:
    """Same iteration-level scheduling as ContinuousBatchEngine (M3): evict
    a sequence the instant it finishes, admit the next queued prompt into
    the freed slot without waiting for the whole batch to drain. Unlike
    M3, KV storage is a fixed-size block pool (BlockAllocator) instead of
    each sequence owning its own growing tensor: every decode step gathers
    each active sequence's tokens from its blocks (gather_and_batch), runs
    the unmodified HF forward(), then scatters only the new token into the
    pool (scatter_new_token) -- old tokens are never rewritten, unlike
    M3's pad_and_batch_caches/split_batched_cache round trip.

    Admission allocates ceil(prompt_len / block_size) blocks up front; if
    the pool doesn't have enough free blocks for the queue's next prompt,
    it simply stays queued and is retried next global step -- no progress
    to lose there. An *active* sequence needing a fresh block mid-decode is
    different: if every active sequence simply waited whenever blocks ran
    short, multiple sequences landing on a block boundary at once with an
    empty pool would deadlock (nobody advances, so nobody ever finishes to
    free blocks). Instead each decode step resolves block pressure by
    recompute-based preemption: evict another active sequence (preferring
    one that doesn't itself need a block this step), freeing all its
    blocks, discarding its progress, and returning it to the front of the
    queue to re-prefill later -- vLLM's actual default policy under memory
    pressure. See docs/superpowers/specs/2026-07-24-m4-paged-kv-cache-design.md
    for why this was chosen over simply reserving each sequence's full
    prompt+max_new_tokens budget upfront (that alternative avoids the
    deadlock too, but reintroduces the reserve-for-worst-case memory waste
    paged attention exists to eliminate).
    """

    def __init__(
        self,
        model,
        tokenizer,
        num_blocks: int,
        block_size: int = 16,
        verbose: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        config = model.config
        num_layers = config.num_hidden_layers
        kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        device = next(model.parameters()).device
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
            num_layers=num_layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=next(model.parameters()).dtype,
            device=device,
        )

    def _prefill(self, prompt_input_ids: torch.Tensor):
        out = self.model(prompt_input_ids, use_cache=True)
        num_layers = len(out.past_key_values.layers)
        keys = [out.past_key_values.layers[l].keys for l in range(num_layers)]
        values = [out.past_key_values.layers[l].values for l in range(num_layers)]
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        return keys, values, next_token

    @torch.inference_mode()
    def generate(
        self,
        prompt_input_ids: list[torch.Tensor],
        max_concurrent_slots: int,
        max_new_tokens: int = 256,
    ) -> ContinuousBatchGenerationResult:
        device = next(self.model.parameters()).device
        eos_token_id = self.tokenizer.eos_token_id
        n = len(prompt_input_ids)
        block_size = self.allocator.block_size

        for ids in prompt_input_ids:
            prompt_len = ids.shape[-1]
            num_blocks_needed = (prompt_len + block_size - 1) // block_size
            if num_blocks_needed > self.allocator.num_blocks:
                raise RuntimeError(
                    f"prompt of length {prompt_len} needs {num_blocks_needed} blocks, "
                    f"more than the pool's total {self.allocator.num_blocks}"
                )

        queue = list(range(n))
        token_ids: list[list[int]] = [[] for _ in range(n)]
        finished_at_step = [-1] * n
        admitted_at_step = [-1] * n
        active: list[_PagedActiveSequence] = []
        global_step = 0

        def admit(seq_id: int, step: int) -> _PagedActiveSequence | None:
            ids = prompt_input_ids[seq_id].to(device)
            block_ids = alloc_blocks_for_length(self.allocator, ids.shape[-1])
            table = SequenceBlockTable(block_ids=block_ids)
            keys, values, next_token = self._prefill(ids)
            write_prefill_tokens(self.allocator, table, keys, values)
            admitted_at_step[seq_id] = step
            tok = next_token.item()
            if tok == eos_token_id:
                finished_at_step[seq_id] = 0
                self.allocator.free(table.block_ids)
                return None
            token_ids[seq_id].append(tok)
            return _PagedActiveSequence(
                seq_id=seq_id, table=table, next_token=next_token, admitted_at_step=step
            )

        def try_admit(step: int) -> None:
            # Peek before popping: if the front of the queue needs more
            # blocks than are currently free, break rather than pop, so it
            # stays queued and gets retried next global step once
            # something frees up. When `active` is empty this can only
            # succeed (num_free() == num_blocks then, and every prompt was
            # validated above to fit within the whole pool), so this never
            # stalls with sequences permanently stuck behind an empty
            # active set.
            while queue and len(active) < max_concurrent_slots:
                seq_id = queue[0]
                prompt_len = prompt_input_ids[seq_id].shape[-1]
                num_blocks_needed = (prompt_len + block_size - 1) // block_size
                if num_blocks_needed > self.allocator.num_free():
                    break
                queue.pop(0)
                seq = admit(seq_id, step)
                if seq is not None:
                    active.append(seq)

        t0 = time.perf_counter()
        try_admit(global_step)

        while active:
            step_active = list(active)  # stable snapshot: next_ids' row i is step_active[i]
            tables = [seq.table for seq in step_active]
            batched_cache, attention_mask, position_ids = gather_and_batch(self.allocator, tables)
            step_attention_mask = torch.cat(
                [attention_mask, torch.ones(len(step_active), 1, dtype=attention_mask.dtype, device=device)],
                dim=-1,
            )
            step_input = torch.cat([seq.next_token for seq in step_active], dim=0)

            out = self.model(
                step_input,
                attention_mask=step_attention_mask,
                position_ids=position_ids,
                past_key_values=batched_cache,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            next_ids = logits.argmax(dim=-1, keepdim=True)

            num_layers = len(out.past_key_values.layers)
            new_keys = [out.past_key_values.layers[l].keys[:, :, -1:, :] for l in range(num_layers)]
            new_values = [out.past_key_values.layers[l].values[:, :, -1:, :] for l in range(num_layers)]

            # Resolve block pressure before scattering: figure out which
            # sequences need a fresh block this step (their length lands
            # exactly on a block boundary -- the same condition
            # scatter_new_token checks internally). If the pool can't
            # cover all of them, preempt other active sequences one at a
            # time -- preferring ones that don't themselves need a block
            # this step, since freeing those costs nothing extra -- until
            # it can. `would_be_free` tracks the free-block count as if
            # each preemption-so-far had already happened (freeing a
            # sequence returns *all* its blocks, not just one, so this
            # usually resolves in a single preemption).
            needs_block = {
                seq.seq_id for seq in step_active if seq.table.length % block_size == 0
            }
            preempted: set[int] = set()
            would_be_free = self.allocator.num_free()
            while len(needs_block - preempted) > would_be_free:
                candidates = [s for s in step_active if s.seq_id not in preempted]
                if len(candidates) <= 1:
                    raise RuntimeError(
                        f"pool of {self.allocator.num_blocks} blocks (block_size="
                        f"{block_size}) cannot fit even a single sequence's growth "
                        "-- increase num_blocks"
                    )
                non_growth = [s for s in candidates if s.seq_id not in needs_block]
                victim = non_growth[-1] if non_growth else candidates[-1]
                preempted.add(victim.seq_id)
                would_be_free += len(victim.table.block_ids)

            for seq in step_active:
                if seq.seq_id in preempted:
                    self.allocator.free(seq.table.block_ids)
                    token_ids[seq.seq_id] = []
                    admitted_at_step[seq.seq_id] = -1
                    queue.insert(0, seq.seq_id)

            global_step += 1
            if self.verbose and global_step == 1:
                print(
                    f"[global step {global_step}] active={len(step_active)} "
                    f"step_input={tuple(step_input.shape)} "
                    f"kv_pool_free={self.allocator.num_free()}/{self.allocator.num_blocks}"
                )

            still_active = []
            for i, seq in enumerate(step_active):
                if seq.seq_id in preempted:
                    continue  # already freed/re-queued above; discard this step's token
                seq.next_token = next_ids[i : i + 1]
                seq.local_step += 1
                tok = next_ids[i, 0].item()
                is_eos = tok == eos_token_id
                # Same max_new_tokens accounting convention as
                # ContinuousBatchEngine (M3) -- see its fix for the
                # max_new_tokens=1 edge case.
                hit_budget = seq.local_step >= max_new_tokens - 1
                if not is_eos and seq.local_step < max_new_tokens:
                    token_ids[seq.seq_id].append(tok)
                # Guaranteed to succeed: the preemption pass above ensured
                # enough free blocks exist for every surviving sequence
                # that needs one this step.
                scatter_new_token(
                    self.allocator,
                    [seq.table],
                    [k[i : i + 1] for k in new_keys],
                    [v[i : i + 1] for v in new_values],
                )
                if is_eos or hit_budget:
                    finished_at_step[seq.seq_id] = seq.local_step
                    self.allocator.free(seq.table.block_ids)
                else:
                    still_active.append(seq)
            active[:] = still_active

            try_admit(global_step)

        wall_clock_s = time.perf_counter() - t0
        total_tokens = sum(max(len(g) - 1, 0) for g in token_ids)
        decode_tok_per_s = (
            total_tokens / wall_clock_s if wall_clock_s > 0 and total_tokens > 0 else 0.0
        )

        texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in token_ids]
        return ContinuousBatchGenerationResult(
            token_ids=token_ids,
            texts=texts,
            finished_at_step=finished_at_step,
            admitted_at_step=admitted_at_step,
            total_global_steps=global_step,
            wall_clock_s=wall_clock_s,
            decode_tok_per_s=decode_tok_per_s,
        )
