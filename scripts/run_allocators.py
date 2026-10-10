"""
run_allocators.py - fair allocator benchmark on a trained MatCCA checkpoint.

  python scripts/run_allocators.py --ckpt runs/s_matcca_strat/ckpt.pt --budgets 32 24 16 12 8 \
      --calib-batches 8 --eval-batches 32 --out results/alloc_s_seed0.json

Budgets are total latent heads summed over layers (S: 8 layers x 8 heads = 64 at full level).
Calibration windows come from the TRAINING shards (with their own seed); scoring uses the fixed
validation windows, so no allocator is scored on the data it was tuned on. Repeat with
--calib-seed 1 2 to report variation across calibration draws.
"""
import argparse
import contextlib
import json
import os

import torch

from matcca.allocate import run_all
from matcca.data import TrainWindows, ValWindows, expand
from matcca.model import GPT, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budgets", nargs="+", type=int, required=True)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--calib-seed", type=int, default=12345)
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--out", default="results/alloc.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = GPT(ModelConfig.from_dict(cfg["model"])).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    if dev == "cuda":
        model.half()  # inference only; allocation and scoring in fp16 on T4
    tw = TrainWindows(expand(cfg["train_shards"]), cfg["seq_len"], data_seed=args.calib_seed)
    calib = [tuple(t.to(dev) for t in tw.batch(i * args.batch, args.batch)) for i in range(args.calib_batches)]
    val = ValWindows(cfg["val_shard"], cfg["seq_len"], args.eval_batches * args.batch * cfg["seq_len"])
    held = [(x.to(dev), y.to(dev)) for x, y in val.batches(args.batch)]
    out = run_all(model, calib, held, args.budgets, include_greedy=not args.no_greedy)
    out.update(ckpt=args.ckpt, calib_seed=args.calib_seed, calib_tokens=args.calib_batches * args.batch * cfg["seq_len"],
               eval_tokens=len(held) * args.batch * cfg["seq_len"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    for B, res in out["budgets"].items():
        print(f"budget {B}:", {k: round(v.get("loss", v.get("mean", float("nan"))), 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
