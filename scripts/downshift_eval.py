"""
downshift_eval.py - quality cost of shrinking caches mid-sequence (the in-flight elasticity claim).

  python scripts/downshift_eval.py --ckpt runs/s_matcca_strat/ckpt.pt --splits 256 512 768 --low 4 2 \
      --batches 64 --out results/downshift_s.json

For each split point and low level: loss on the tokens after the split when the model (a) stays at
the full level, (b) runs at the low level from the start, (c) runs full, then truncates every cache
to the low level at the split. (c) close to (b) means a server can downshift running requests
under memory pressure instead of preempting them.
"""
import argparse
import json
import os

import torch

from matcca.data import ValWindows
from matcca.evaluate import downshift_losses
from matcca.model import GPT, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", type=int, default=[256, 512, 768])
    ap.add_argument("--low", nargs="+", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--out", default="results/downshift.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = GPT(ModelConfig.from_dict(cfg["model"])).to(dev)
    model.load_state_dict(st["model"])
    amp = torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else None
    val = ValWindows(cfg["val_shard"], cfg["seq_len"], args.batches * args.batch * cfg["seq_len"])
    rows = []
    for s in args.splits:
        for lo in args.low:
            r = downshift_losses(model, val, args.batch, split=s, low=lo, device=dev, amp_ctx=amp)
            r.update(split=s, low=lo)
            print(json.dumps(r))
            rows.append(r)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
