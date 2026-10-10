"""
bench_decode.py - measured decode throughput and memory for each config (fp16, real KV cache).
Weights are random: speed and memory do not depend on training. Run on the T4 you train on.

  python scripts/bench_decode.py --configs configs/s/mha.json configs/s/cca4.json configs/s/matcca_strat.json \
      --batches 1 8 32 128 256 --prompt 1024 --gen 128 --out results/decode_s.json

For nested configs every level in nest_levels is benchmarked. Also writes, per config, the largest
batch that fits under --cache-gb of KV cache at --context tokens (capacity at a memory cap).
"""
import argparse
import json
import os

import torch

from matcca.evaluate import decode_benchmark
from matcca.model import GPT, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--cache-gb", type=float, default=12.0)
    ap.add_argument("--out", default="results/decode.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    results = []
    for path in args.configs:
        cfg = json.load(open(path))
        mc = ModelConfig.from_dict(cfg["model"])
        mc.max_seq_len = max(mc.max_seq_len, args.prompt + args.gen, args.context)
        model = GPT(mc).to(dev, dtype)
        for lv in (cfg.get("nest_levels") or [None]):
            per_tok = model.cache_bytes_per_token(lv)
            cap = int(args.cache_gb * 2**30 // (per_tok * args.context))
            rows = decode_benchmark(model, args.batches, args.prompt, args.gen, levels=lv, device=dev, dtype=dtype)
            for r in rows:
                r.update(config=cfg["run_name"], level=lv, cache_bytes_per_token=per_tok,
                         max_batch_at_cap=cap, cap_gb=args.cache_gb, cap_context=args.context)
                print(json.dumps(r))
            results.extend(rows)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
