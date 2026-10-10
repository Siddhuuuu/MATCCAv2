"""
audit_tests.py — independent checks of the MatCCA code exactly as it produced the paper
(the %%writefile versions inside MatCCA.ipynb, not the stale top-level .py files).

Run:  python -I audit_tests.py <path-to-nbcode-dir>
"""
import sys, math, itertools
sys.path.insert(0, sys.argv[1])
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attention import CompressedConvolutionalAttention
from tiny_transformer import TinyTransformerLM

torch.manual_seed(0)
torch.set_grad_enabled(False)
results = {}

# ---------------------------------------------------------------------------
# T1. Initial loss with the paper's config and the default PyTorch init
# ---------------------------------------------------------------------------
def init_loss(model, vocab, T=128, B=4):
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    _, loss = model(x, targets=y)
    return loss.item()

def gpt2_init(model, n_layers):
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, mean=0.0, std=0.02)
        if name.endswith(("out_proj.weight", "up_proj.weight", "net.2.weight")):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        if "bias" in name:
            nn.init.zeros_(p)

for kind, kw in [("mha", {}), ("cca", dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4))]:
    torch.manual_seed(0)
    m = TinyTransformerLM(vocab_size=8000, dim=512, n_layers=8, n_heads=8, max_seq_len=512,
                          attn_kind=kind, attn_kwargs=kw).eval()
    l_default = init_loss(m, 8000)
    emb_std = m.tok_emb.weight.std().item()
    gpt2_init(m, 8)
    l_fixed = init_loss(m, 8000)
    print(f"[T1] {kind}: init loss default={l_default:.1f} nats (tok_emb std={emb_std:.2f}), "
          f"GPT-2-style init={l_fixed:.2f} nats, ln(8000)={math.log(8000):.2f}")
    results[f"T1_{kind}"] = (l_default, l_fixed)

# ---------------------------------------------------------------------------
# T2. Matryoshka nesting consistency of the value path (value-shift layout)
#     Set up_proj to an identity read-out so the module output exposes the
#     per-head attention outputs, then compare heads shared by r=128 and r=64.
# ---------------------------------------------------------------------------
torch.manual_seed(0)
E, C, H = 512, 4, 8
cca = CompressedConvolutionalAttention(E, n_heads_latent=H, n_kv_heads_latent=H,
                                       compression_q=C, compression_kv=C).eval()
r_max, hd = cca.latent_dim_q_max, cca.head_dim_q
W = torch.zeros(E, r_max); W[:r_max, :r_max] = torch.eye(r_max)
cca.up_proj.weight.copy_(W)
x = torch.randn(2, 64, E)
full = cca(x)[..., :r_max].reshape(2, 64, H, hd)
for r in (64, 32):
    h = r // hd
    part = cca(x, active_rank_q=r, active_rank_kv=r)[..., :r].reshape(2, 64, h, hd)
    diffs = [(full[:, :, i] - part[:, :, i]).abs().max().item() for i in range(h)]
    print(f"[T2] rank {r} ({h} heads): max |out_head(r) - out_head(full)| per shared head = "
          + ", ".join(f"h{i}:{d:.2e}" for i, d in enumerate(diffs)))
    results[f"T2_r{r}"] = diffs

# Which value source feeds each head at each rank (the root cause)
def v_source_map(r):
    half = r // 2
    src = [f"now[{i}:{i+hd}]" for i in range(0, half, hd)] + [f"prev[{i}:{i+hd}]" for i in range(0, half, hd)]
    return src
print("[T2] value source per head, r=128:", v_source_map(128))
print("[T2] value source per head, r=64 :", v_source_map(64))
print("[T2] value source per head, r=32 :", v_source_map(32))

# ---------------------------------------------------------------------------
# T3. Allocator evaluation: what ranks were actually evaluated?
#     Reproduces allocation_to_per_layer_ranks() from water_filling_vs_greedy_real.py
# ---------------------------------------------------------------------------
def rounded(alloc, head_dim=16):
    return [max(head_dim, round(a / head_dim) * head_dim) for a in alloc]

published = {  # raw allocations printed in the final notebook run (cell 50) = paper Appendix E
    ("2x", 512): {"WF": [128, 128, 111, 65, 35, 21, 14, 10], "OWF": [128, 128, 111, 65, 35, 21, 13, 11],
                  "Greedy": [128, 128, 128, 32, 16, 16, 32, 32]},
    ("4x", 256): {"WF": [102, 83, 33, 16, 9, 6, 4, 3], "OWF": [107, 82, 32, 16, 9, 5, 3, 2],
                  "Greedy": [64, 96, 16, 16, 16, 16, 16, 16]},
    ("8x", 128): {"WF": [48, 42, 19, 9, 6, 3, 1, 0], "OWF": [52, 40, 19, 9, 5, 2, 1, 0],
                  "Greedy": [16] * 8},
}
for (lvl, budget), algs in published.items():
    for name, a in algs.items():
        ev = rounded(a) if name != "Greedy" else a
        tot = sum(ev)
        print(f"[T3] {lvl} budget={budget}: {name:6s} raw={a} sum={sum(a)} -> evaluated={ev} "
              f"sum={tot} ({100 * (tot - budget) / budget:+.1f}% vs budget)")
        results[f"T3_{lvl}_{name}"] = (tot, budget)

# ---------------------------------------------------------------------------
# T4. Greedy feasibility at "8x": min rank per layer = head_dim = 16, 8 layers, budget 128
# ---------------------------------------------------------------------------
levels = range(16, 129, 16)
feasible = [c for c in itertools.product(levels, repeat=8) if sum(c) == 128]
print(f"[T4] allocations with every layer in {{16,...,128}} (step 16) summing to 128: {len(feasible)} -> {feasible}")
feasible_256 = sum(1 for c in itertools.product(levels, repeat=8) if sum(c) == 256)
print(f"[T4] same count at budget 256: {feasible_256}")

# ---------------------------------------------------------------------------
# T5. Statistics on the paper's per-seed numbers (Appendix D)
# ---------------------------------------------------------------------------
from scipy import stats
D = {
    "MHA": [3.228, 3.173, 3.156], "GQA": [3.215, 3.216, 3.206], "MLA": [3.467, 3.385, 3.467],
    "CCA4": [3.096, 3.059, 3.040], "CCA8": [3.113, 3.082, 3.118], "CCA16": [3.154, 3.150, 3.138],
    "MatCCA@4": [3.200, 3.283, 3.252], "MatCCA@8": [3.228, 3.332, 3.286], "MatCCA@16": [3.290, 3.376, 3.382],
}
def ci95(v):
    v = np.array(v); m = v.mean(); s = v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * s / math.sqrt(len(v))
    return m, s, m - h, m + h
for k, v in D.items():
    m, s, lo, hi = ci95(v)
    print(f"[T5] {k:10s} mean={m:.4f} sd={s:.4f} 95% CI=[{lo:.3f}, {hi:.3f}]")
for a, b in [("MHA", "CCA16"), ("MHA", "CCA8"), ("MHA", "CCA4"), ("GQA", "CCA16"), ("CCA16", "CCA8"), ("CCA8", "CCA4"),
             ("MatCCA@4", "CCA4"), ("MatCCA@16", "CCA16"), ("MHA", "MatCCA@4")]:
    w = stats.ttest_ind(D[a], D[b], equal_var=False)
    p = stats.ttest_rel(D[a], D[b])
    print(f"[T5] {a} - {b}: diff={np.mean(D[a]) - np.mean(D[b]):+.4f}  Welch p={w.pvalue:.3f}  paired p={p.pvalue:.3f}")
# +-1 sd intervals as the paper reports them
m1, s1 = np.mean(D["MHA"]), np.std(D["MHA"], ddof=1); m2, s2 = np.mean(D["CCA16"]), np.std(D["CCA16"], ddof=1)
print(f"[T5] +-1sd: MHA [{m1 - s1:.3f}, {m1 + s1:.3f}] vs CCA16 [{m2 - s2:.3f}, {m2 + s2:.3f}] -> overlap={m1 - s1 < m2 + s2}")

# ---------------------------------------------------------------------------
# T6. Re-evaluation noise visible in the notebook logs (same checkpoint, re-run estimate_loss)
# ---------------------------------------------------------------------------
reeval = {"MLA seed0": (3.4647, 3.4668), "MLA seed1": (3.3874, 3.3845), "CCA4 seed1": (3.0491, 3.0588),
          "CCA8 seed1": (3.0605, 3.0824), "MatCCA seed2 (final vs level-1.00 eval)": (3.2329, 3.252)}
for k, (a, b) in reeval.items():
    print(f"[T6] {k}: {a:.4f} vs {b:.4f}  |diff|={abs(a - b):.4f}")

# ---------------------------------------------------------------------------
# T7. Throughput / MFU implied by the training logs
# ---------------------------------------------------------------------------
tokens = 5000 * 32 * 512
for name, n_params, minutes in [("MHA", 29.56e6, 97.7), ("CCA4", 23.41e6, 101.0), ("MLA", 26.94e6, 88.4)]:
    tps = tokens / (minutes * 60)
    # 6N per token for weights (embedding lookups excluded is approx; tied head included) + attention 6*L*T*d
    flops = 6 * n_params + 6 * 8 * 512 * 512
    tf = tps * flops / 1e12
    print(f"[T7] {name}: {tps:,.0f} tok/s, ~{tf:.1f} TFLOP/s achieved (T4 peak: 8.1 fp32, 65 fp16 tensor)")
print("[T7] total tokens per run:", f"{tokens / 1e6:.1f}M", "= epochs over 43.6M:", f"{tokens / 43.6e6:.2f}")
