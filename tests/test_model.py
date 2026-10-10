"""Model correctness tests (CPU, fp32). Run: python -m pytest -q tests"""
import math
import pytest
import torch

from matcca.model import GPT, ModelConfig

torch.manual_seed(0)
V = 97

CONFIGS = {
    "mha": dict(attn="mha"),
    "gqa": dict(attn="gqa", n_kv_heads=2),
    "mla": dict(attn="mla", mla_kv_rank=24, mla_rope_dim=8),
    "cca": dict(attn="cca", cca_heads=4, cca_head_dim=8),
    "cca_default_conv": dict(attn="cca", cca_heads=4, cca_head_dim=8, cca_conv_init="default"),
    "cca_k2": dict(attn="cca", cca_heads=4, cca_head_dim=8, cca_conv_kernel=2),
    "cca_noconv": dict(attn="cca", cca_heads=4, cca_head_dim=8, cca_conv=False),
}


def make(kind, n_layers=2, **kw):
    cfg = ModelConfig(vocab_size=V, d_model=64, n_layers=n_layers, n_heads=4, max_seq_len=64,
                      **{**CONFIGS[kind], **kw})
    m = GPT(cfg).eval()
    if CONFIGS[kind].get("attn") == "cca":
        # make conv / temperature non-trivial so tests exercise them
        with torch.no_grad():
            for b in m.blocks:
                a = b.attn
                a.key_temp.normal_(0, 0.3)
                if a.use_conv:
                    for p in (a.conv_q_w0, a.conv_q_w1, a.conv_k_w0, a.conv_k_w1, a.conv_q_b0, a.conv_k_b1):
                        p.add_(0.2 * torch.randn_like(p))
    return m


def test_init_loss_close_to_uniform():
    m = GPT(ModelConfig(vocab_size=1000, d_model=128, n_layers=4, n_heads=4, attn="cca",
                        cca_heads=4, cca_head_dim=16)).eval()
    x = torch.randint(0, 1000, (4, 64))
    _, loss = m(x, targets=torch.randint(0, 1000, (4, 64)))
    assert abs(loss.item() - math.log(1000)) < 0.3, loss.item()


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_causality(kind):
    m = make(kind)
    x = torch.randint(0, V, (2, 32))
    x2 = x.clone(); x2[:, 20:] = torch.randint(0, V, (2, 12))
    with torch.no_grad():
        a, _ = m(x); b, _ = m(x2)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5)
    assert (a[:, 20:] - b[:, 20:]).abs().max() > 1e-3


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_decode_matches_full_forward(kind):
    m = make(kind)
    x = torch.randint(0, V, (2, 24))
    levels = [3, 2] if kind.startswith("cca") else None
    with torch.no_grad():
        full, _ = m(x, levels=levels)
        caches = m.new_caches(2, 24, levels=levels)
        outs = [m(x[:, :10], levels=levels, caches=caches, start_pos=0)[0]]      # prefill
        outs.append(m(x[:, 10:16], levels=levels, caches=caches, start_pos=10)[0])  # chunked prefill
        for t in range(16, 24):                                                  # token by token
            outs.append(m(x[:, t:t + 1], levels=levels, caches=caches, start_pos=t)[0])
    inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


@pytest.mark.parametrize("kind", ["cca", "cca_default_conv", "cca_k2", "cca_noconv"])
def test_lower_level_is_exact_subnetwork(kind):
    """Level k must equal the full model with heads >= k switched off (gates = 0)."""
    m = make(kind, n_layers=3)
    x = torch.randint(0, V, (2, 20))
    with torch.no_grad():
        for k in (1, 2, 3):
            a, _ = m(x, levels=k)
            gates = [torch.cat([torch.ones(k), torch.zeros(4 - k)]) for _ in range(3)]
            b, _ = m(x, gates=gates)
            assert torch.allclose(a, b, atol=1e-5), (k, (a - b).abs().max())


@pytest.mark.parametrize("kind", ["cca", "cca_default_conv", "cca_noconv"])
def test_inflight_cache_truncation_single_layer(kind):
    """For one layer, decoding at full level, truncating the cache to k heads and continuing at level
    k gives exactly what level k would have produced from the start (no recomputation needed)."""
    m = make(kind, n_layers=1)
    x = torch.randint(0, V, (2, 30))
    with torch.no_grad():
        ref, _ = m(x, levels=2)
        caches = m.new_caches(2, 30)
        m(x[:, :18], caches=caches, start_pos=0)          # full level: 4 heads
        caches[0].truncate(2)
        outs = []
        for t in range(18, 30):
            outs.append(m(x[:, t:t + 1], levels=2, caches=caches, start_pos=t)[0])
    inc = torch.cat(outs, dim=1)
    assert torch.allclose(ref[:, 18:], inc, atol=1e-4), (ref[:, 18:] - inc).abs().max()


def test_per_layer_levels_and_level_zero():
    m = make("cca", n_layers=3)
    x = torch.randint(0, V, (2, 16))
    with torch.no_grad():
        a, _ = m(x, levels=[4, 0, 1])
        gates = [torch.ones(4), torch.zeros(4), torch.tensor([1.0, 0, 0, 0])]
        b, _ = m(x, gates=gates)
    assert torch.allclose(a, b, atol=1e-5)


def test_cache_bytes():
    m = make("cca", n_layers=2)
    assert m.cache_bytes_per_token() == 2 * (2 * 4 * 8 * 2)
    assert m.cache_bytes_per_token(levels=[2, 1]) == 2 * 2 * 8 * 2 + 2 * 1 * 8 * 2
    mha = make("mha")
    assert mha.cache_bytes_per_token() == 2 * (2 * 4 * 16 * 2)


def test_chunked_loss_matches_full():
    cfg = ModelConfig(vocab_size=V, d_model=64, n_layers=2, n_heads=4, max_seq_len=64, attn="mha")
    m = GPT(cfg)
    x = torch.randint(0, V, (2, 32)); y = torch.randint(0, V, (2, 32))
    _, l1 = m(x, targets=y)
    m.cfg.loss_chunk = 7
    m.train()
    _, l2 = m(x, targets=y)
    assert torch.allclose(l1, l2, atol=1e-5)
    l2.backward()
    g = m.wte.weight.grad.clone()
    m.zero_grad(); m.cfg.loss_chunk = 0
    _, l3 = m(x, targets=y); l3.backward()
    assert torch.allclose(g, m.wte.weight.grad, atol=1e-5)


def test_downshift_eval_single_layer_equals_low(tmp_path):
    import numpy as np
    from matcca.data import ValWindows, write_shard
    from matcca.evaluate import downshift_losses
    write_shard(str(tmp_path / "v.bin"), np.random.default_rng(0).integers(0, V, 4000))
    val = ValWindows(str(tmp_path / "v.bin"), 32, 32 * 8)
    m = make("cca", n_layers=1)
    r = downshift_losses(m, val, 4, split=20, low=2)
    assert abs(r["downshift"] - r["low"]) < 1e-5, r
    m2 = make("cca", n_layers=3)
    r2 = downshift_losses(m2, val, 4, split=20, low=2)
    assert all(np.isfinite(v) for v in r2.values())
