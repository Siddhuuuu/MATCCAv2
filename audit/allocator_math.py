"""
allocator_math.py — checks the assumptions behind water-filling (WF) and output-weighted
WF (OWF) on a small MatCCA trained on CPU (char-level Shakespeare, the corpus shipped in
the repo). Uses the notebook's attention.py / tiny_transformer.py unchanged.

  A. Variance kept by the truncation MatCCA actually applies (first r coordinates)
     vs. the variance WF assumes it keeps (top-r eigen-directions).
  B. Per-layer K-latent variance scale: the global sort compares raw variances across
     layers in units that differ by layer.
  C. OWF is not invariant to an exact symmetry of the model (permuting heads inside the
     v_now block): same function, same K eigenvalues, different OWF scores.

Run:  python -I allocator_math.py <path-to-nbcode-dir>
"""
import sys, math, random, time
sys.path.insert(0, sys.argv[1])
import os
import torch, torch.nn as nn
import numpy as np
from tiny_transformer import TinyTransformerLM
from real_data import prepare_dataset, get_batch

torch.manual_seed(0); random.seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
train, val, tok = prepare_dataset(os.path.join(sys.argv[1], "shakespeare.txt"))
V, DIM, L, T, B = tok.vocab_size, 128, 4, 128, 32
kw = dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4)
m = TinyTransformerLM(V, dim=DIM, n_layers=L, n_heads=8, max_seq_len=T, attn_kind="cca", attn_kwargs=kw)
for n, p in m.named_parameters():
    if p.dim() >= 2: nn.init.normal_(p, 0, 0.02)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.0)
r_max = DIM // 4; hd = r_max // 8
levels = [r_max, r_max // 2, r_max // 4]
t0 = time.time()
for step in range(600):
    x, y = get_batch(train, B, T)
    r = random.choice(levels)
    _, loss = m(x, targets=y, active_rank_q=r, active_rank_kv=r)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 200 == 0 or step == 599:
        print(f"step {step} r={r} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
m.eval()

# capture pre-conv K latents (what WF uses)
for blk in m.blocks: blk.attn.capture_activations = True; blk.attn._captured_k_lat.clear()
with torch.no_grad():
    for _ in range(8):
        x, _ = get_batch(val, 16, T); m(x)
print("\nA/B: per layer, fraction of K-latent variance kept at rank r")
print("     'eig' = top-r eigen-directions (what WF's distortion assumes)")
print("     'coord' = first r coordinates (what MatCCA's truncation keeps)")
evs = {}
for i, blk in enumerate(m.blocks):
    A = torch.cat(blk.attn._captured_k_lat).reshape(-1, r_max).double()
    A = A - A.mean(0, keepdim=True)
    cov = A.T @ A / A.shape[0]
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    evs[i] = ev.numpy()
    tot = ev.sum().item()
    row = []
    for r in (8, 16):
        eig_keep = ev[:r].sum().item() / tot
        coord_keep = torch.diagonal(cov)[:r].sum().item() / tot
        row.append(f"r={r}: eig {eig_keep:.2f} vs coord {coord_keep:.2f}")
    print(f"  layer {i}: total variance {tot:.4f} | " + " | ".join(row))
    blk.attn.capture_activations = False

# global WF on these eigenvalues
from water_filling import reverse_waterfilling_allocation
alloc, theta, _ = reverse_waterfilling_allocation({f"layer_{i}": evs[i] for i in evs}, L * r_max // 4)
print(f"  WF allocation at budget {L * r_max // 4}: {[alloc[f'layer_{i}'] for i in range(L)]} (theta={theta:.4g})")

# C. OWF symmetry test on layer 0: permute heads inside the v_now block (heads 0..3)
att = m.blocks[0].attn
perm_heads = [3, 2, 1, 0, 4, 5, 6, 7]
idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in perm_heads])
x, _ = get_batch(val, 4, T)
with torch.no_grad():
    h_in = m.blocks[0].norm1(m.tok_emb(x) + m.pos_emb(torch.arange(T))[None])
    out_before = att(h_in)
    for c in (att.conv_q, att.conv_k):
        c.conv0.weight.copy_(c.conv0.weight[idx]); c.conv0.bias.copy_(c.conv0.bias[idx])
        c.conv1.weight.copy_(c.conv1.weight[idx]); c.conv1.bias.copy_(c.conv1.bias[idx])
    att.down_q.weight.copy_(att.down_q.weight[idx]); att.down_k.weight.copy_(att.down_k.weight[idx])
    att.key_temp.copy_(att.key_temp[torch.tensor(perm_heads)])
    half = r_max // 2
    idx_now = idx[idx < half]
    att.down_v_now.weight.copy_(att.down_v_now.weight[idx_now])
    att.up_proj.weight.copy_(att.up_proj.weight[:, idx])
    out_after = att(h_in)
print(f"\nC: max |output change| after permuting heads 0-3 of layer 0: {(out_before - out_after).abs().max().item():.2e} (same function)")
att.capture_activations = True; att._captured_k_lat.clear()
with torch.no_grad(): att(h_in)
A = att._captured_k_lat[0].reshape(-1, r_max).double(); A = A - A.mean(0)
ev_after = torch.linalg.eigvalsh(A.T @ A / A.shape[0]).flip(0)
att._captured_k_lat.clear(); att.capture_activations = False
W_before_norms = None
print("   K eigenvalues are permutation-invariant; OWF scores = sorted eigenvalue_i x ||W_up[:, i]||^2 by coordinate i:")
with torch.no_grad():
    norms_after = (att.up_proj.weight ** 2).sum(0)
    norms_before = norms_after[torch.argsort(idx)]
ev_before_t = torch.tensor(evs[0])
s_before = (ev_before_t * norms_before.double()).numpy()
s_after = (ev_before_t * norms_after.double()).numpy()
print(f"   top-8 OWF score positions before: {np.argsort(-s_before)[:8].tolist()}")
print(f"   top-8 OWF score positions after : {np.argsort(-s_after)[:8].tolist()}")
print(f"   max |score change| = {np.abs(s_before - s_after).max():.3e}  (scores change for an identical model)")
