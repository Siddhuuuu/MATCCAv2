"""
evaluate.py - deterministic validation loss and decode benchmarking.
"""
from __future__ import annotations

import contextlib
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import ValWindows
from .model import GPT


@torch.no_grad()
def val_loss(model: GPT, val: ValWindows, batch_size: int, levels=None, device="cpu",
             amp_ctx=None, rank: int = 0, world: int = 1) -> float:
    """Mean per-token cross-entropy (nats) over the fixed validation windows. Identical windows for
    every model and every call, so differences between runs are never evaluation-sampling noise."""
    was_training = model.training
    model.eval()
    ctx = amp_ctx if amp_ctx is not None else contextlib.nullcontext()
    tot = torch.zeros(2, dtype=torch.float64, device=device)
    for x, y in val.batches(batch_size, rank, world):
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, _ = model(x, levels=levels)
        tot[0] += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum").double()
        tot[1] += y.numel()
    if world > 1:
        dist.all_reduce(tot)
    if was_training:
        model.train()
    return (tot[0] / tot[1]).item()


def val_losses(model, val, batch_size, levels_list: Sequence, **kw) -> Dict[str, float]:
    return {str(lv): val_loss(model, val, batch_size, levels=lv, **kw) for lv in levels_list}


@torch.no_grad()
def downshift_losses(model: GPT, val: ValWindows, batch_size: int, split: int, low, high=None,
                     device="cpu", amp_ctx=None, max_batches: Optional[int] = None) -> Dict[str, float]:
    """Quality of shrinking a running request's cache mid-sequence (MatCCA in-flight downshift).

    For each validation window, the first `split` tokens are processed at level `high` (None = full),
    then every layer's cache is truncated to level `low` and the remaining tokens are processed at
    `low` (one chunk, exact equivalent of token-by-token decoding). Loss is measured on tokens after
    `split` only, and compared with running at `high` throughout and at `low` from the start."""
    model.eval()
    ctx = amp_ctx if amp_ctx is not None else contextlib.nullcontext()
    sums = {"high": 0.0, "low": 0.0, "downshift": 0.0}
    n = 0
    for bi, (x, y) in enumerate(val.batches(batch_size)):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        B, T = x.shape
        tail = y[:, split:].reshape(-1)
        with ctx:
            for key, lv in (("high", high), ("low", low)):
                logits, _ = model(x, levels=lv)
                sums[key] += F.cross_entropy(logits[:, split:].float().reshape(-1, logits.size(-1)), tail,
                                             reduction="sum").item()
            caches = model.new_caches(B, T, device=device, levels=high)
            model(x[:, :split], levels=high, caches=caches, start_pos=0)
            lows = low if isinstance(low, (list, tuple)) else [low] * len(caches)
            for c, k in zip(caches, lows):
                c.truncate(model.blocks[0].attn.H if k is None else k)
            logits, _ = model(x[:, split:], levels=low, caches=caches, start_pos=split)
            sums["downshift"] += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tail,
                                                 reduction="sum").item()
        n += tail.numel()
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def decode_benchmark(model: GPT, batch_sizes: List[int], prompt_len: int, gen_len: int,
                     levels=None, device="cuda", dtype=torch.float16, warmup: int = 2) -> List[dict]:
    """Prefill + token-by-token decode with a real cache. Reports measured peak memory, cache bytes,
    prefill time and decode tokens/s. Use a model already moved to `device` and cast to `dtype`."""
    model.eval()
    V = model.cfg.vocab_size
    out = []
    for B in batch_sizes:
        def run():
            caches = model.new_caches(B, prompt_len + gen_len, device=device, dtype=dtype, levels=levels)
            x = torch.randint(0, V, (B, prompt_len), device=device)
            sync()
            t0 = time.perf_counter()
            logits, _ = model(x, levels=levels, caches=caches, start_pos=0)
            sync()
            t1 = time.perf_counter()
            nxt = logits[:, -1:].argmax(-1)
            for i in range(gen_len):
                logits, _ = model(nxt, levels=levels, caches=caches, start_pos=prompt_len + i)
                nxt = logits[:, -1:].argmax(-1)
            sync()
            t2 = time.perf_counter()
            return caches, t1 - t0, t2 - t1

        def sync():
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()

        try:
            for _ in range(warmup):
                run()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            caches, t_prefill, t_decode = run()
            peak = torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0
            out.append(dict(batch=B, prompt_len=prompt_len, gen_len=gen_len,
                            prefill_s=t_prefill, prefill_tok_s=B * prompt_len / t_prefill,
                            decode_s=t_decode, decode_tok_s=B * gen_len / t_decode,
                            ms_per_step=1000 * t_decode / gen_len,
                            cache_bytes=sum(c.nbytes() for c in caches), peak_bytes=peak, oom=False))
            del caches
        except torch.cuda.OutOfMemoryError:
            out.append(dict(batch=B, prompt_len=prompt_len, gen_len=gen_len, oom=True))
            torch.cuda.empty_cache()
    return out
