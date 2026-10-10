import numpy as np
import pytest
import torch

from matcca.allocate import (brute_force, dp_allocate, fisher_costs, greedy_kl, profile_costs,
                             random_allocations, run_all, uniform, variance_costs)
from matcca.model import GPT, ModelConfig


@pytest.mark.parametrize("seed", range(5))
def test_dp_is_exact(seed):
    rng = np.random.default_rng(seed)
    L, H = 4, 3
    # non-convex, arbitrary costs on purpose
    costs = rng.normal(size=(L, H + 1)); costs[:, H] = 0
    for budget in range(0, L * H + 1):
        a, v = dp_allocate(costs, budget)
        b, w = brute_force(costs, budget)
        assert sum(a) == budget and abs(v - w) < 1e-9


def test_uniform_and_random_hit_budget():
    assert sum(uniform(8, 8, 20)) == 20
    for a in random_allocations(8, 8, 20, 50, seed=1):
        assert sum(a) == 20 and all(0 <= k <= 8 for k in a)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=61, d_model=32, n_layers=3, n_heads=4, max_seq_len=32, attn="cca",
                      cca_heads=4, cca_head_dim=8)
    m = GPT(cfg).eval()
    mk = lambda s: [(torch.randint(0, 61, (2, 16), generator=torch.Generator().manual_seed(s + i)),
                     torch.randint(0, 61, (2, 16), generator=torch.Generator().manual_seed(100 + s + i))) for i in range(3)]
    return m, mk(0), mk(50)


def test_cost_matrices_and_budget(tiny):
    m, calib, held = tiny
    for c in (variance_costs(m, calib), fisher_costs(m, calib), profile_costs(m, calib)[0]):
        assert c.shape == (3, 5)
        a, _ = dp_allocate(c, 6)
        assert sum(a) == 6
    g, passes = greedy_kl(m, calib, 5)
    assert sum(g) == 5 and passes > 0


def test_run_all_exact_budgets(tiny):
    m, calib, held = tiny
    out = run_all(m, calib, held, budgets=[3, 6, 9], n_random=5)
    for B, res in out["budgets"].items():
        for name, r in res.items():
            if name != "random":
                assert sum(r["alloc"]) == int(B), (name, r["alloc"], B)
