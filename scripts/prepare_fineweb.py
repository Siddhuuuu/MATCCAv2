"""
prepare_fineweb.py - download pre-tokenized GPT-2 FineWeb shards (100M tokens, ~200 MB each) from
the Hugging Face dataset kjj0/fineweb10B-gpt2 (modded-nanogpt / llm.c format).

  python scripts/prepare_fineweb.py --train-shards 10        # 1.0B tokens: enough for an S run
  python scripts/prepare_fineweb.py --train-shards 25        # 2.5B tokens: an M run

Only the shards a run needs; files already present are skipped. Inside Kaggle this takes about a
minute per 3 GB.
"""
import argparse
import os

from huggingface_hub import hf_hub_download

REPO = "kjj0/fineweb10B-gpt2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", type=int, default=10)
    ap.add_argument("--out", default="data/fineweb10B")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, args.train_shards + 1)]
    for n in names:
        if not os.path.exists(os.path.join(args.out, n)):
            hf_hub_download(repo_id=REPO, filename=n, repo_type="dataset", local_dir=args.out)
        print("ok", n)


if __name__ == "__main__":
    main()
