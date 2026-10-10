"""Training smoke test and exact-resume test on CPU with synthetic shards."""
import json
import os

import numpy as np
import torch

from matcca.data import TrainWindows, ValWindows, write_shard
from matcca.train import load_config, train


def make_shards(tmp, vocab=50, n=60_000):
    rng = np.random.default_rng(0)
    # learnable structure: repeating random motifs
    motifs = rng.integers(0, vocab, size=(40, 12))
    seq = np.concatenate([motifs[rng.integers(0, 40)] for _ in range(n // 12)])
    write_shard(os.path.join(tmp, "train_000.bin"), seq[: n // 2])
    write_shard(os.path.join(tmp, "train_001.bin"), seq[n // 2: n - 5000])
    write_shard(os.path.join(tmp, "val_000.bin"), seq[n - 5000:])


def base_cfg(tmp, run, **kw):
    d = dict(run_name=run, out_dir=os.path.join(tmp, "runs"),
             model=dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32, attn="cca",
                        cca_heads=4, cca_head_dim=8),
             train_shards=os.path.join(tmp, "train_*.bin"), val_shard=os.path.join(tmp, "val_000.bin"),
             seq_len=32, micro_batch=4, grad_accum=3, max_steps=40, lr=3e-3, eval_every=20,
             val_tokens=2048, eval_batch=8, log_every=10, amp="none", ckpt_every_min=1e9,
             nest_mode="stratified", nest_levels=[4, 2, 1])
    d.update(kw)
    path = os.path.join(tmp, f"{run}.json")
    with open(path, "w") as f:
        json.dump(d, f)
    return load_config(path, [])


def test_data_is_deterministic_and_paired(tmp_path):
    make_shards(str(tmp_path))
    paths = sorted(str(p) for p in tmp_path.glob("train_*.bin"))
    a = TrainWindows(paths, 32, data_seed=3).batch(1000, 8)[0]
    b = TrainWindows(paths, 32, data_seed=3).batch(1000, 8)[0]
    c = TrainWindows(paths, 32, data_seed=4).batch(1000, 8)[0]
    assert torch.equal(a, b) and not torch.equal(a, c)
    v = ValWindows(str(tmp_path / "val_000.bin"), 32, 2048)
    assert sum(x.shape[0] for x, _ in v.batches(8)) == 2048 // 32


def test_training_reduces_loss(tmp_path):
    make_shards(str(tmp_path))
    res = train(base_cfg(str(tmp_path), "smoke", max_steps=150, lr=1e-2))
    vals = res["final_val"]
    assert set(vals) == {"4", "2", "1"}
    assert vals["4"] < 3.0 and vals["1"] < 3.3, vals  # ln(64) = 4.16 at init


def test_resume_is_exact(tmp_path):
    make_shards(str(tmp_path))
    for mode in ("random", "stratified", "sum"):
        straight = base_cfg(str(tmp_path), f"straight_{mode}", nest_mode=mode)
        train(straight)
        part = base_cfg(str(tmp_path), f"resumed_{mode}", nest_mode=mode, time_limit_min=1e-9)
        train(part)          # stops after the first step (time limit), checkpoints
        resumed = base_cfg(str(tmp_path), f"resumed_{mode}", nest_mode=mode)
        train(resumed)       # continues from the checkpoint
        a = torch.load(os.path.join(straight.out_dir, straight.run_name, "ckpt.pt"), weights_only=False)["model"]
        b = torch.load(os.path.join(resumed.out_dir, resumed.run_name, "ckpt.pt"), weights_only=False)["model"]
        for k in a:
            assert torch.equal(a[k], b[k]), (mode, k)
