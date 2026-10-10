"""
stats.py - the statistics the paper should report: t-intervals, paired and Welch tests, bootstrap.
With 3 seeds, a 95% t-interval is mean +/- 2.48 sd, so report intervals, not +/- 1 sd.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

try:
    from scipy import stats as _st
except Exception:  # scipy optional
    _st = None


def t_interval(x: Sequence[float], conf: float = 0.95) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    n, m = len(x), float(np.mean(x))
    if n < 2:
        return dict(mean=m, lo=float("nan"), hi=float("nan"), sd=float("nan"), n=n)
    sd = float(np.std(x, ddof=1))
    tq = _st.t.ppf(0.5 + conf / 2, n - 1) if _st else {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 1.96)
    h = tq * sd / math.sqrt(n)
    return dict(mean=m, lo=m - h, hi=m + h, sd=sd, n=n)


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """a, b: per-seed values with the SAME data seeds (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(diff=float(d.mean()), **{f"ci_{k}": v for k, v in t_interval(d).items() if k in ("lo", "hi")})
    if _st and len(d) > 1:
        out["p_paired"] = float(_st.ttest_rel(a, b).pvalue)
        out["p_welch"] = float(_st.ttest_ind(a, b, equal_var=False).pvalue)
    return out


def bootstrap_ci(x: Sequence[float], n_boot: int = 10_000, conf: float = 0.95, seed: int = 0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    lo, hi = np.quantile(means, [(1 - conf) / 2, (1 + conf) / 2])
    return float(lo), float(hi)
