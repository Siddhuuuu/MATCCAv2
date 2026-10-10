"""
train.py - one training run from a JSON config, on one GPU or with DDP.

  python -m matcca.train --config configs/s/cca4.json
  python -m matcca.train --config configs/s/cca4.json --set seed=1 data_seed=1 compile=true
  torchrun --nproc_per_node=2 -m matcca.train --config configs/m/matcca_strat.json

What it does that v1 did not:
  * fp16 autocast + GradScaler on T4 (bf16 where supported), optional torch.compile, fused AdamW
  * warmup + cosine schedule, gradient clipping, no weight decay on norms / convs / temperatures
  * deterministic data order (data_seed) shared by all architectures -> paired comparisons
  * checkpoints with optimizer, scaler, every RNG state and the step, written atomically,
    by wall-clock interval and before a session time limit (Kaggle: 12 h); exact resume
  * nested training modes for MatCCA: random | stratified | sum (see NESTING below)
  * evaluation on the full, fixed validation windows at every nesting level
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import TrainWindows, ValWindows, expand
from .evaluate import val_losses
from .model import GPT, ModelConfig

# NESTING
#   none        : ordinary training (all non-CCA baselines, dedicated CCA)
#   random      : one level per optimizer step, sampled with a seeded RNG (v1 behaviour, reproducible)
#   stratified  : micro-batch m of step s runs level levels[(s * accum + m) % n]; same cost as random,
#                 every level is updated in (almost) every step, lower gradient variance
#   sum         : every micro-batch runs all levels, loss = weighted mean (MRL objective, n x cost)


@dataclass
class TrainConfig:
    run_name: str = "run"
    out_dir: str = "runs"
    model: dict = field(default_factory=dict)
    train_shards: str = "data/fineweb10B/fineweb_train_*.bin"
    val_shard: str = "data/fineweb10B/fineweb_val_000000.bin"
    seq_len: int = 1024
    micro_batch: int = 16
    grad_accum: int = 16
    global_batch_tokens: int = 0       # >0: grad_accum = this / (micro_batch * seq_len * world), so the
                                       # optimizer sees the same batch on 1 or 2 GPUs
    total_tokens: int = 1_000_000_000
    max_steps: int = 0                 # 0 -> total_tokens / tokens_per_step
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 0                      # model init and level sampling
    data_seed: int = 0                 # data order; keep it equal across architectures
    eval_every: int = 250
    val_tokens: int = 10_000_000
    eval_batch: int = 16
    log_every: int = 10
    ckpt_every_min: float = 30.0
    time_limit_min: float = 0.0        # >0: checkpoint and exit after this many minutes
    amp: str = "auto"                  # auto | fp16 | bf16 | none
    compile: bool = False
    nest_mode: str = "none"            # none | random | stratified | sum
    nest_levels: List[int] = field(default_factory=list)
    nest_weights: List[float] = field(default_factory=list)
    hub_repo: str = ""                 # optional private Hugging Face repo for checkpoints


# --------------------------------------------------------------------------------------
def load_config(path: Optional[str], overrides: List[str]) -> TrainConfig:
    d = {}
    if path:
        with open(path) as f:
            d = json.load(f)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        if k.startswith("model."):
            d.setdefault("model", {})[k[6:]] = v
        else:
            d[k] = v
    fields = TrainConfig.__dataclass_fields__
    unknown = [k for k in d if k not in fields]
    assert not unknown, f"unknown config keys: {unknown}"
    return TrainConfig(**d)


def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank, local, world = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
        torch.cuda.set_device(local)
        return True, rank, world, torch.device("cuda", local)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, dev


def amp_setup(mode: str, device):
    if device.type != "cuda" or mode == "none":
        return contextlib.nullcontext(), None
    if mode == "auto":
        # is_bf16_supported() is True on a T4 (software emulation, very slow, and torch.compile skips it),
        # so decide by hardware: native bf16 needs compute capability >= 8.0 (Ampere or newer).
        mode = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    dt = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dt), dt


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig, device):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 and "conv_" not in n else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8)
    if device.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(groups, **kw)


def lr_at(step: int, max_steps: int, cfg: TrainConfig) -> float:
    warm = max(1, int(cfg.warmup_frac * max_steps))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, max_steps - warm))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def level_plan(cfg: TrainConfig, step: int, rng: np.random.Generator):
    """Per micro-batch: ("single", level) or ("multi", [levels])."""
    A, lv = cfg.grad_accum, cfg.nest_levels
    if cfg.nest_mode == "none" or not lv:
        return [("single", None)] * A
    if cfg.nest_mode == "random":
        l = int(lv[rng.integers(len(lv))])
        return [("single", l)] * A
    if cfg.nest_mode == "stratified":
        return [("single", int(lv[(step * A + m) % len(lv)])) for m in range(A)]
    if cfg.nest_mode == "sum":
        return [("multi", [int(x) for x in lv])] * A
    raise ValueError(cfg.nest_mode)


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def hub_push(cfg: TrainConfig, path: str):
    if not cfg.hub_repo:
        return
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=path, path_in_repo=f"{cfg.run_name}/{os.path.basename(path)}",
                            repo_id=cfg.hub_repo, repo_type="model")
    except Exception as e:  # never kill a run over an upload
        print(f"[hub] upload failed: {e}")


def hub_pull(cfg: TrainConfig, path: str):
    if not cfg.hub_repo or os.path.exists(path):
        return
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(cfg.hub_repo, f"{cfg.run_name}/{os.path.basename(path)}", repo_type="model",
                            local_dir=os.path.dirname(path))
        if os.path.abspath(p) != os.path.abspath(path):
            os.replace(p, path)
        print(f"[hub] pulled checkpoint for {cfg.run_name}")
    except Exception as e:
        print(f"[hub] no remote checkpoint ({type(e).__name__})")


# --------------------------------------------------------------------------------------
def train(cfg: TrainConfig) -> dict:
    ddp, rank, world, device = setup_dist()
    master = rank == 0
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    log_path = os.path.join(run_dir, "log.jsonl")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    level_rng = np.random.default_rng([cfg.seed, 777])

    mcfg = ModelConfig.from_dict(cfg.model)
    mcfg.max_seq_len = max(mcfg.max_seq_len, cfg.seq_len)
    raw = GPT(mcfg).to(device)
    opt = make_optimizer(raw, cfg, device)
    amp_ctx, amp_dtype = amp_setup(cfg.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    if cfg.global_batch_tokens:
        per = cfg.micro_batch * cfg.seq_len * world
        assert cfg.global_batch_tokens % per == 0, "global_batch_tokens must divide by micro_batch*seq_len*world"
        cfg.grad_accum = cfg.global_batch_tokens // per
    tokens_per_step = cfg.micro_batch * cfg.seq_len * cfg.grad_accum * world
    max_steps = cfg.max_steps or max(1, cfg.total_tokens // tokens_per_step)

    start_step, evals, elapsed_prev = 0, [], 0.0
    if master:
        hub_pull(cfg, ckpt_path)
    if ddp:
        dist.barrier()
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_step, evals, elapsed_prev = st["step"], st.get("evals", []), st.get("elapsed", 0.0)
        level_rng.bit_generator.state = st["level_rng"]
        torch.set_rng_state(st["torch_rng"].cpu())
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        if master:
            print(f"resumed {cfg.run_name} at step {start_step}/{max_steps}")
        if st.get("done"):
            return st.get("results", {})

    model = raw
    if cfg.compile:
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[device.index])

    train_data = TrainWindows(expand(cfg.train_shards), cfg.seq_len, cfg.data_seed)
    val = ValWindows(cfg.val_shard, cfg.seq_len, cfg.val_tokens)
    eval_levels = [None] if cfg.nest_mode == "none" or not cfg.nest_levels else [int(l) for l in cfg.nest_levels]

    def log(rec):
        if master:
            print(json.dumps(rec))
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def checkpoint(step_next: int, done: bool = False, results: Optional[dict] = None):
        if not master:
            return
        st = dict(model=raw.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step_next,
                  evals=evals, elapsed=elapsed_prev + time.time() - t_start, cfg=asdict(cfg),
                  level_rng=level_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                  py_rng=random.getstate(), np_rng=np.random.get_state(), done=done, results=results)
        atomic_save(st, ckpt_path)
        hub_push(cfg, ckpt_path)

    if master:
        log(dict(event="start", run=cfg.run_name, params=raw.n_params(), max_steps=max_steps,
                 tokens_per_step=tokens_per_step, world=world, amp=str(amp_dtype), device=str(device),
                 cache_bytes_per_token=raw.cache_bytes_per_token()))

    t_start = time.time()
    last_ckpt = time.time()
    t_log, tok_log, flops_log = time.time(), 0, 0.0
    step = start_step
    model.train()
    while step < max_steps:
        lr = lr_at(step, max_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        plan = level_plan(cfg, step, level_rng)
        loss_acc = 0.0
        for m, (kind, lv) in enumerate(plan):
            offset = ((step * cfg.grad_accum + m) * world + rank) * cfg.micro_batch
            x, y = train_data.batch(offset, cfg.micro_batch)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync_ctx = model.no_sync() if (ddp and m < len(plan) - 1) else contextlib.nullcontext()
            with sync_ctx:
                with amp_ctx:
                    if kind == "multi":
                        _, loss = model(x, targets=y, multi_levels=lv, multi_weights=cfg.nest_weights or None)
                    else:
                        _, loss = model(x, targets=y, levels=lv)
                scaler.scale(loss / cfg.grad_accum).backward()
            loss_acc += loss.item() / cfg.grad_accum
            levels_run = lv if kind == "multi" else [lv]
            flops_log += sum(raw.flops_per_token(cfg.seq_len, l) for l in levels_run) * x.numel() * world
            tok_log += x.numel() * world
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0 or step == max_steps:
            dt = time.time() - t_log
            log(dict(step=step, tokens=step * tokens_per_step, loss=round(loss_acc, 5), lr=lr, gnorm=round(gnorm, 4),
                     tok_s=round(tok_log / dt), tflops=round(flops_log / dt / 1e12, 2), scale=scaler.get_scale()))
            t_log, tok_log, flops_log = time.time(), 0, 0.0
        if step % cfg.eval_every == 0 or step == max_steps:
            res = val_losses(raw, val, cfg.eval_batch, eval_levels, device=device, amp_ctx=amp_ctx,
                             rank=rank, world=world)
            evals.append(dict(step=step, tokens=step * tokens_per_step, val=res))
            log(dict(event="eval", step=step, tokens=step * tokens_per_step, val=res))
        now = time.time()
        over_time = cfg.time_limit_min > 0 and (elapsed_prev + now - t_start) / 60 > cfg.time_limit_min
        if (now - last_ckpt) / 60 > cfg.ckpt_every_min or over_time:
            checkpoint(step)
            last_ckpt = now
            if over_time:
                log(dict(event="paused", step=step))
                if ddp:
                    dist.destroy_process_group()
                return {}

    results = dict(run=cfg.run_name, params=raw.n_params(), steps=step, tokens=step * tokens_per_step,
                   final_val=evals[-1]["val"] if evals else {}, cache_bytes_per_token=raw.cache_bytes_per_token(),
                   train_hours=(elapsed_prev + time.time() - t_start) / 3600, cfg=asdict(cfg))
    checkpoint(step, done=True, results=results)
    if master:
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    if ddp:
        dist.destroy_process_group()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides: key=value or model.key=value")
    args = ap.parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
