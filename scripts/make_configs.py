"""
make_configs.py - writes the experiment matrix as JSON configs (configs/s/*.json, configs/m/*.json).

Scales (GPT-2 tokenizer, vocab padded to 50,304, seq 1,024):
  S: d 512, 8 layers, 8 heads  ~51M params, 1.0B tokens  (20 tokens/param)
  M: d 768, 12 layers, 12 heads ~124M params, 2.5B tokens

Equal-cache pairs (fp16 bytes per token per layer): GQA with H/4 KV heads = CCA 4x (S: 512 B, M: 768 B);
MLA with latent d/4 - 32 plus a 32-dim RoPE key = CCA 8x (S: 256 B, M: 384 B).
Run every config with the same data_seed for a given seed index (paired comparisons).
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

SCALES = {
    "s": dict(d_model=512, n_layers=8, n_heads=8, tokens=1_000_000_000, micro=16, gbt=262_144, lr=1e-3),
    "m": dict(d_model=768, n_layers=12, n_heads=12, tokens=2_500_000_000, micro=8, gbt=524_288, lr=6e-4),
}


def variants(d, H):
    dh = d // H
    return {
        "mha": dict(attn="mha"),
        "gqa": dict(attn="gqa", n_kv_heads=H // 4),                        # same cache as CCA 4x
        "mla": dict(attn="mla", mla_kv_rank=d // 4 - dh // 2, mla_rope_dim=dh // 2),  # same cache as CCA 8x
        "cca1x": dict(attn="cca", cca_heads=H, cca_head_dim=dh),          # no compression: isolates convs
        "cca4": dict(attn="cca", cca_heads=d // 64, cca_head_dim=16),      # 4x: r = d/4
        "cca8": dict(attn="cca", cca_heads=d // 128, cca_head_dim=16),     # 8x, drop heads (MatCCA geometry)
        "cca8_narrow": dict(attn="cca", cca_heads=d // 64, cca_head_dim=8),  # 8x, narrow heads (v1 geometry)
        "cca16": dict(attn="cca", cca_heads=d // 256, cca_head_dim=16),
        "cca4_noconv": dict(attn="cca", cca_heads=d // 64, cca_head_dim=16, cca_conv=False),
        "cca4_noqkmean": dict(attn="cca", cca_heads=d // 64, cca_head_dim=16, cca_qk_mean=False),
        "cca4_novshift": dict(attn="cca", cca_heads=d // 64, cca_head_dim=16, cca_v_shift=False),
    }


def nested(d):
    H = d // 64
    levels = [H, H // 2, H // 4]                     # 4x, 8x, 16x relative to MHA
    base = dict(attn="cca", cca_heads=H, cca_head_dim=16)
    return {
        "matcca_strat": (base, dict(nest_mode="stratified", nest_levels=levels)),
        "matcca_random": (base, dict(nest_mode="random", nest_levels=levels)),
        "matcca_sum": (base, dict(nest_mode="sum", nest_levels=levels)),   # 3x cost per step
    }


def write(scale, name, model, extra, s):
    cfg = dict(run_name=f"{scale}_{name}", out_dir="runs",
               model=dict(vocab_size=50304, max_seq_len=1024, d_model=s["d_model"], n_layers=s["n_layers"],
                          n_heads=s["n_heads"], **model),
               train_shards="data/fineweb10B/fineweb_train_*.bin",
               val_shard="data/fineweb10B/fineweb_val_000000.bin",
               seq_len=1024, micro_batch=s["micro"], global_batch_tokens=s["gbt"], total_tokens=s["tokens"],
               lr=s["lr"], warmup_frac=0.02, min_lr_ratio=0.1, weight_decay=0.1, grad_clip=1.0,
               eval_every=250, val_tokens=10_000_000, eval_batch=16, ckpt_every_min=30,
               time_limit_min=11 * 60, amp="auto", compile=True, seed=0, data_seed=0)
    cfg.update(extra)
    os.makedirs(os.path.join(ROOT, scale), exist_ok=True)
    with open(os.path.join(ROOT, scale, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    for scale, s in SCALES.items():
        for name, model in variants(s["d_model"], s["n_heads"]).items():
            write(scale, name, model, {}, s)
        for name, (model, extra) in nested(s["d_model"]).items():
            write(scale, name, model, extra, s)
    print("wrote configs to", os.path.abspath(ROOT))
