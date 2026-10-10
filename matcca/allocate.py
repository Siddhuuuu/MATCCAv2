"""
allocate.py - per-layer head allocation for a MatCCA model under an EXACT budget.

Every allocator returns k_l in {0, ..., H} heads per layer with sum_l k_l == budget, so all of
them are compared on the same feasible set and the same KV-cache size. (v1 rounded water-filling
allocations after the fact, which exceeded the budget by up to 50%, and gave greedy a different
minimum rank.)

Allocators
  uniform      as even as possible
  random       uniformly random feasible allocations (the spread to beat)
  variance     per-head energy of k (post-conv) and v that would be dropped; cheap, but the scale is
               not comparable across layers (kept as the corrected version of v1's water-filling)
  fisher_dp    loss-increase estimate per head from a gate's gradient and Fisher term, then exact DP
  profile_dp   measured loss increase with one layer truncated at a time, then exact DP
  greedy_kl    remove one head at a time where KL to the full model grows least (MatryoshkaKV-style)

Calibrate on one set of batches, score on a disjoint held-out set (`score_allocations`).
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT, CCAttention


def _heads(model: GPT) -> Tuple[int, int]:
    attn = [b.attn for b in model.blocks]
    assert all(isinstance(a, CCAttention) for a in attn), "allocation needs a CCA / MatCCA model"
    return len(attn), attn[0].H


# --------------------------------------------------------------------------------------
# exact solver
# --------------------------------------------------------------------------------------
def dp_allocate(costs: np.ndarray, budget: int) -> Tuple[List[int], float]:
    """Minimize sum_l costs[l, k_l] subject to sum_l k_l == budget, k_l in {0..H}.
    costs[l, k] = loss increase when layer l keeps k heads. Exact multiple-choice knapsack DP."""
    L, H1 = costs.shape
    assert 0 <= budget <= L * (H1 - 1)
    best = np.full((L + 1, budget + 1), np.inf)
    best[0, 0] = 0.0
    choice = np.zeros((L, budget + 1), dtype=np.int64)
    for l in range(L):
        for b in range(budget + 1):
            if not np.isfinite(best[l, b]):
                continue
            for k in range(min(H1, budget - b + 1)):
                v = best[l, b] + costs[l, k]
                if v < best[l + 1, b + k]:
                    best[l + 1, b + k] = v
                    choice[l, b + k] = k
    ks, b = [], budget
    for l in reversed(range(L)):
        k = int(choice[l, b]); ks.append(k); b -= k
    return ks[::-1], float(best[L, budget])


def brute_force(costs: np.ndarray, budget: int):
    L, H1 = costs.shape
    best, arg = np.inf, None
    for ks in itertools.product(range(H1), repeat=L):
        if sum(ks) == budget:
            v = sum(costs[l, k] for l, k in enumerate(ks))
            if v < best:
                best, arg = v, list(ks)
    return arg, best


# --------------------------------------------------------------------------------------
# simple allocators
# --------------------------------------------------------------------------------------
def uniform(L: int, H: int, budget: int) -> List[int]:
    base, rem = divmod(budget, L)
    assert base + (rem > 0) <= H
    return [base + (1 if l < rem else 0) for l in range(L)]


def random_allocations(L: int, H: int, budget: int, n: int, seed: int = 0) -> List[List[int]]:
    """Uniform over all feasible allocations (counted by DP, sampled backwards)."""
    rng = np.random.default_rng(seed)
    ways = np.zeros((L + 1, budget + 1), dtype=object)
    ways[L, 0] = 1
    for l in reversed(range(L)):
        for b in range(budget + 1):
            ways[l, b] = sum(ways[l + 1, b - k] for k in range(min(H, b) + 1))
    out = []
    for _ in range(n):
        ks, b = [], budget
        for l in range(L):
            w = np.array([float(ways[l + 1, b - k]) if k <= b else 0.0 for k in range(H + 1)])
            k = int(rng.choice(H + 1, p=w / w.sum())); ks.append(k); b -= k
        out.append(ks)
    return out


# --------------------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------------------
@torch.no_grad()
def mean_loss(model: GPT, batches, levels, amp_ctx=None) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        if amp_ctx is not None:
            with amp_ctx:
                logits, _ = model(x, levels=levels)
        else:
            logits, _ = model(x, levels=levels)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return tot / n


@torch.no_grad()
def _kl(model, xs, ref_logp, levels) -> float:
    tot, n = 0.0, 0
    for x, lp_ref in zip(xs, ref_logp):
        lp = F.log_softmax(model(x, levels=levels)[0].float(), -1)
        tot += (lp_ref.exp() * (lp_ref - lp)).sum().item()
        n += x.numel()
    return tot / n


def costs_from_head_deltas(delta: np.ndarray) -> np.ndarray:
    """delta[l, h] = estimated loss increase from removing head h (alone). Keeping k heads removes
    heads k..H-1 (nesting order), so cost[l, k] = sum_{h >= k} delta[l, h]."""
    L, H = delta.shape
    cost = np.zeros((L, H + 1))
    for k in range(H + 1):
        cost[:, k] = delta[:, k:].sum(1)
    return cost


# --------------------------------------------------------------------------------------
# allocators that look at the model
# --------------------------------------------------------------------------------------
@torch.no_grad()
def variance_costs(model: GPT, calib) -> np.ndarray:
    L, H = _heads(model)
    model.eval()
    model.set_collect_stats(True)
    for x, _ in calib:
        model(x)
    model.set_collect_stats(False)
    delta = np.zeros((L, H))
    for l, b in enumerate(model.blocks):
        s = b.attn.stats
        delta[l] = ((s["k"] + s["v"]) / s["n"]).cpu().numpy()
    return costs_from_head_deltas(delta)


def fisher_costs(model: GPT, calib) -> np.ndarray:
    """Second-order Taylor estimate of the loss increase when a layer keeps only its first k heads
    (gates of heads k..H-1 go from 1 to 0):

        cost[l, k] = -sum_{h>=k} grad[l,h] + 1/2 * sum_{h,h'>=k} F_l[h,h']

    F_l is the empirical Fisher of the per-token loss for layer l's gates, estimated from batch
    gradients as N_tokens * E_b[g_b g_b^T] (exact if per-token gradients are uncorrelated; validate
    against profile_costs on one checkpoint). Cost: one forward + backward per calibration batch."""
    L, H = _heads(model)
    model.eval()
    dev = next(model.parameters()).device
    g_sum = torch.zeros(L, H, device=dev, dtype=torch.float64)
    G_sum = torch.zeros(L, H, H, device=dev, dtype=torch.float64)
    nb, ntok = 0, 0
    for x, y in calib:
        gates = torch.ones(L, H, device=dev, requires_grad=True)
        _, loss = model(x, targets=y, gates=[gates[l] for l in range(L)])
        (g,) = torch.autograd.grad(loss, gates)
        g = g.double()
        g_sum += g
        G_sum += g[:, :, None] * g[:, None, :]
        nb += 1
        ntok = y.numel()
    grad = (g_sum / nb).cpu().numpy()
    fisher = (ntok * G_sum / nb).cpu().numpy()
    cost = np.zeros((L, H + 1))
    for l in range(L):
        for k in range(H + 1):
            cost[l, k] = -grad[l, k:].sum() + 0.5 * fisher[l, k:, k:].sum()
    return cost


@torch.no_grad()
def profile_costs(model: GPT, calib) -> Tuple[np.ndarray, int]:
    L, H = _heads(model)
    base = mean_loss(model, calib, None)
    cost = np.zeros((L, H + 1))
    passes = len(calib)
    for l in range(L):
        for k in range(H):
            lv = [H] * L
            lv[l] = k
            cost[l, k] = mean_loss(model, calib, lv) - base
            passes += len(calib)
    return cost, passes


@torch.no_grad()
def greedy_kl(model: GPT, calib, budget: int) -> Tuple[List[int], int]:
    L, H = _heads(model)
    model.eval()
    xs = [x for x, _ in calib]
    ref = [F.log_softmax(model(x)[0].float(), -1) for x in xs]
    cur, passes = [H] * L, len(xs)
    while sum(cur) > budget:
        best = None
        for l in range(L):
            if cur[l] == 0:
                continue
            cand = cur.copy(); cand[l] -= 1
            kl = _kl(model, xs, ref, cand); passes += len(xs)
            if best is None or kl < best[0]:
                best = (kl, cand)
        cur = best[1]
    return cur, passes


# --------------------------------------------------------------------------------------
def run_all(model: GPT, calib, held_out, budgets: Sequence[int], n_random: int = 20,
            include_greedy: bool = True, include_profile: bool = True) -> Dict:
    """Run every allocator at every budget; score each allocation on held_out. Costs include the
    calibration passes each allocator needs (no "zero forward passes" accounting)."""
    L, H = _heads(model)
    out = {"L": L, "H": H, "budgets": {}}
    t = time.time(); var_c = variance_costs(model, calib); t_var = time.time() - t
    t = time.time(); fis_c = fisher_costs(model, calib); t_fis = time.time() - t
    if include_profile:
        t = time.time(); prof_c, prof_passes = profile_costs(model, calib); t_prof = time.time() - t
    for B in budgets:
        res = {}
        def add(name, alloc, seconds, passes):
            assert sum(alloc) == B and all(0 <= k <= H for k in alloc), (name, alloc, B)
            res[name] = dict(alloc=list(map(int, alloc)), loss=mean_loss(model, held_out, alloc),
                             seconds=seconds, forward_passes=passes)
        if B % L == 0 or B // L + 1 <= H:
            add("uniform", uniform(L, H, B), 0.0, 0)
        rand = [mean_loss(model, held_out, a) for a in random_allocations(L, H, B, n_random)]
        res["random"] = dict(losses=rand, mean=float(np.mean(rand)), min=float(np.min(rand)))
        add("variance", dp_allocate(var_c, B)[0], t_var, len(calib))
        add("fisher_dp", dp_allocate(fis_c, B)[0], t_fis, 2 * len(calib))
        if include_profile:
            add("profile_dp", dp_allocate(prof_c, B)[0], t_prof, prof_passes)
        if include_greedy:
            t = time.time(); g, gp = greedy_kl(model, calib, B)
            add("greedy_kl", g, time.time() - t, gp)
        out["budgets"][str(B)] = res
    return out
