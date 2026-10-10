"""
model.py - GPT backbone with interchangeable attention: MHA / GQA, MLA and CCA (MatCCA).

Every attention module supports
  * training over full sequences (causal),
  * incremental decoding with an explicit cache (prefill, chunked prefill, one token at a time).

CCA additionally supports exact Matryoshka nesting over latent heads:
  * `level` = number of active latent heads, global or per layer (0 = attention off),
  * a model run at level k is an exact sub-network of the full model (fixed head layout,
    including the value-shift heads), and
  * a running request's cache can be truncated in place to fewer heads (`CCACache.truncate`).

Fixes relative to the v1 code (see README "What changed"):
  * N(0, 0.02) init with residual projections scaled by 1/sqrt(2L) (v1 started at ~330 nats),
  * value-shift sources fixed per head (even heads: current token, odd heads: previous token),
    so truncation drops whole (q, k, v) heads instead of reassigning value sources,
  * RoPE in every attention type instead of learned absolute positions,
  * MLA with the latent RMSNorm and decoupled RoPE key used by DeepSeek-V2,
  * a real KV cache and decode path, so memory and decode speed can be measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

Level = Optional[Union[int, Sequence[Optional[int]]]]


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 50304
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8                 # query heads for MHA / GQA / MLA
    max_seq_len: int = 1024
    attn: str = "mha"                # "mha" | "gqa" | "mla" | "cca"
    n_kv_heads: Optional[int] = None # GQA key/value heads (None = MHA)
    # MLA (DeepSeek-V2 style, no query compression)
    mla_kv_rank: int = 96            # cached latent width
    mla_rope_dim: int = 32           # decoupled RoPE key width (also cached)
    # CCA / MatCCA
    cca_heads: int = 8               # latent heads at the full level
    cca_head_dim: int = 16           # latent head width (fixed across levels)
    cca_conv_kernel: int = 4
    cca_conv: bool = True
    cca_qk_mean: bool = True
    cca_v_shift: bool = True
    cca_conv_init: str = "identity"  # "identity" (pass-through) | "default" (PyTorch conv init)
    # backbone
    mlp_ratio: int = 4
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    loss_chunk: int = 0              # >0: chunked + checkpointed LM-head loss (saves logits memory)

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        fields = ModelConfig.__dataclass_fields__
        return ModelConfig(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_tables(dim: int, max_len: int, base: float):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D) with interleaved pairs (2i, 2i+1); cos/sin: (T, D/2)."""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s = cos[None, None], sin[None, None]
    y = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return y.to(x.dtype)


def _sdpa(q, k, v, start_pos: int):
    """Causal attention for a query block starting at absolute position start_pos, keys 0..start_pos+Tq-1."""
    Tq, Tk = q.shape[-2], k.shape[-2]
    if Tq == 1:
        return F.scaled_dot_product_attention(q, k, v)
    if Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    qpos = torch.arange(start_pos, start_pos + Tq, device=q.device)[:, None]
    kpos = torch.arange(Tk, device=q.device)[None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=kpos <= qpos)


class Linear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, residual: bool = False):
        super().__init__(d_in, d_out, bias=False)
        self.residual = residual  # residual output projections get a 1/sqrt(2L) init


# --------------------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------------------
class KVCache:
    """Contiguous per-layer cache: k, v of shape (B, H, T_max, D)."""

    def __init__(self, B, H, max_len, dk, dv, device, dtype):
        self.k = torch.zeros(B, H, max_len, dk, device=device, dtype=dtype)
        self.v = torch.zeros(B, H, max_len, dv, device=device, dtype=dtype)
        self.len = 0

    def append(self, k, v):
        T = k.shape[2]
        self.k[:, :, self.len:self.len + T] = k
        self.v[:, :, self.len:self.len + T] = v
        self.len += T
        return self.k[:, :, :self.len], self.v[:, :, :self.len]

    def nbytes(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


class MLACache:
    """Caches the normalized latent c (B, T, r) and the RoPE'd shared key (B, 1, T, dr)."""

    def __init__(self, B, max_len, r, dr, device, dtype):
        self.c = torch.zeros(B, max_len, r, device=device, dtype=dtype)
        self.kr = torch.zeros(B, 1, max_len, dr, device=device, dtype=dtype)
        self.len = 0

    def append(self, c, kr):
        T = c.shape[1]
        self.c[:, self.len:self.len + T] = c
        self.kr[:, :, self.len:self.len + T] = kr
        self.len += T
        return self.c[:, :self.len], self.kr[:, :, :self.len]

    def nbytes(self) -> int:
        return (self.c.numel() + self.kr.numel()) * self.c.element_size()


class CCACache:
    """KV cache for CCA plus the small recurrent state of its convolutions and value shift.

    k, v: (B, H_active, T_max, dh).  Conv states hold the last (K-1) pre-conv inputs and
    conv0 outputs for q and k; v_last holds the previous token's value projection.
    `truncate(h)` drops trailing heads in place (a copy, so the memory is released)."""

    def __init__(self, B, heads, max_len, dh, K, device, dtype):
        r = heads * dh
        self.heads, self.dh = heads, dh
        self.k = torch.zeros(B, heads, max_len, dh, device=device, dtype=dtype)
        self.v = torch.zeros(B, heads, max_len, dh, device=device, dtype=dtype)
        z = lambda: torch.zeros(B, max(K - 1, 0), r, device=device, dtype=dtype)
        self.qx, self.qy0, self.kx, self.ky0 = z(), z(), z(), z()
        self.v_last = torch.zeros(B, 1, r, device=device, dtype=dtype)
        self.len = 0

    def append(self, k, v):
        T = k.shape[2]
        self.k[:, :, self.len:self.len + T] = k
        self.v[:, :, self.len:self.len + T] = v
        self.len += T
        return self.k[:, :, :self.len], self.v[:, :, :self.len]

    def truncate(self, heads: int):
        assert 0 <= heads <= self.heads
        c = heads * self.dh
        self.k = self.k[:, :heads].contiguous()
        self.v = self.v[:, :heads].contiguous()
        self.qx, self.qy0 = self.qx[..., :c].contiguous(), self.qy0[..., :c].contiguous()
        self.kx, self.ky0 = self.kx[..., :c].contiguous(), self.ky0[..., :c].contiguous()
        self.v_last = self.v_last[..., :c].contiguous()
        self.heads = heads

    def nbytes(self) -> int:
        ts = [self.k, self.v, self.qx, self.qy0, self.kx, self.ky0, self.v_last]
        return sum(t.numel() for t in ts) * self.k.element_size()


# --------------------------------------------------------------------------------------
# Attention modules
# --------------------------------------------------------------------------------------
class GQAttention(nn.Module):
    """MHA when n_kv_heads == n_heads, GQA otherwise. RoPE on q and k."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.H, self.Hkv = H, cfg.n_kv_heads or H
        assert d % H == 0 and H % self.Hkv == 0
        self.dh = d // H
        self.q = Linear(d, H * self.dh)
        self.k = Linear(d, self.Hkv * self.dh)
        self.v = Linear(d, self.Hkv * self.dh)
        self.o = Linear(H * self.dh, d, residual=True)
        cos, sin = rope_tables(self.dh, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def new_cache(self, B, max_len, device, dtype, level=None):
        return KVCache(B, self.Hkv, max_len, self.dh, self.dh, device, dtype)

    def cache_bytes_per_token(self, level=None, bytes_per_el=2):
        return 2 * self.Hkv * self.dh * bytes_per_el

    def attn_flops_per_token(self, T, level=None):
        return 6 * T * 2 * self.H * self.dh  # scores + AV, fwd+bwd (full, not causal-halved)

    def forward(self, x, cache=None, start_pos=0, level=None, gates=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.Hkv, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.Hkv, self.dh).transpose(1, 2)
        cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(k, v)
        if self.Hkv != self.H:
            rep = self.H // self.Hkv
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        out = _sdpa(q, k, v, start_pos)
        if gates is not None:
            out = out * gates.view(1, -1, 1, 1).to(out.dtype)
        return self.o(out.transpose(1, 2).reshape(B, T, self.H * self.dh))


class MLAttention(nn.Module):
    """Multi-head latent attention (DeepSeek-V2): RMSNorm on the KV latent, decoupled RoPE key.
    Decode is the non-absorbed form (k, v re-expanded from the cached latent each step), which is
    correct but slower than an absorbed kernel; benchmark numbers for MLA are therefore pessimistic."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.H = H
        dh = d // H
        self.dr = cfg.mla_rope_dim
        self.dn = dh - self.dr
        self.dv = dh
        assert self.dn > 0 and self.dr % 2 == 0
        self.r = cfg.mla_kv_rank
        self.q = Linear(d, H * (self.dn + self.dr))
        self.kv_down = Linear(d, self.r + self.dr)
        self.kv_norm = RMSNorm(self.r, cfg.norm_eps)
        self.kv_up = Linear(self.r, H * (self.dn + self.dv))
        self.o = Linear(H * self.dv, d, residual=True)
        cos, sin = rope_tables(self.dr, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def new_cache(self, B, max_len, device, dtype, level=None):
        return MLACache(B, max_len, self.r, self.dr, device, dtype)

    def cache_bytes_per_token(self, level=None, bytes_per_el=2):
        return (self.r + self.dr) * bytes_per_el

    def attn_flops_per_token(self, T, level=None):
        return 6 * T * self.H * ((self.dn + self.dr) + self.dv)

    def forward(self, x, cache=None, start_pos=0, level=None, gates=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.dn + self.dr).transpose(1, 2)
        q_n, q_r = q.split([self.dn, self.dr], dim=-1)
        c, k_r = self.kv_down(x).split([self.r, self.dr], dim=-1)
        c = self.kv_norm(c)
        cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]
        q_r = apply_rope(q_r, cos, sin)
        k_r = apply_rope(k_r.view(B, 1, T, self.dr), cos, sin)
        if cache is not None:
            c, k_r = cache.append(c, k_r)
        Tk = c.shape[1]
        kv = self.kv_up(c).view(B, Tk, self.H, self.dn + self.dv).transpose(1, 2)
        k_n, v = kv.split([self.dn, self.dv], dim=-1)
        k = torch.cat([k_n, k_r.expand(B, self.H, Tk, self.dr)], dim=-1)
        q = torch.cat([q_n, q_r], dim=-1)
        out = _sdpa(q, k, v, start_pos)
        if gates is not None:
            out = out * gates.view(1, -1, 1, 1).to(out.dtype)
        return self.o(out.transpose(1, 2).reshape(B, T, self.H * self.dv))


class CCAttention(nn.Module):
    """Compressed Convolutional Attention with exact nesting over latent heads.

    Per active head h (h < level):
      q_pre, k_pre, v_all = x W_q[h], x W_k[h], x W_v[h]                (down-projections)
      q_post, k_post = conv1_h(conv0(q_pre)), conv1_h(conv0(k_pre))      (causal; conv0 depthwise,
                                                                          conv1 grouped per head)
      m = (q_pre + k_pre) / 2;  q = q_post + m;  k = k_post + m          (q-k mean)
      q, k <- sqrt(dh) * q/|q|, sqrt(dh) * exp(t_h) * k/|k|; then RoPE
      v_h(t) = v_all_h(t) for even h, v_all_h(t-1) for odd h            (value shift, fixed per head)
      out = sum_h  W_up[:, h] attn_h
    Every quantity of head h depends only on head h's own parameters, so level k is an exact
    sub-network and cached heads can be dropped without recomputation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H, dh, K = cfg.d_model, cfg.cca_heads, cfg.cca_head_dim, cfg.cca_conv_kernel
        assert dh % 2 == 0, "RoPE needs an even head width"
        self.H, self.dh, self.K = H, dh, K
        self.r = H * dh
        self.use_conv, self.use_qk_mean, self.use_v_shift = cfg.cca_conv, cfg.cca_qk_mean, cfg.cca_v_shift
        self.q_down = Linear(d, self.r)
        self.k_down = Linear(d, self.r)
        self.v_down = Linear(d, self.r)
        self.up = Linear(self.r, d, residual=True)
        if self.use_conv:
            self.conv_q_w0 = nn.Parameter(torch.empty(self.r, K))
            self.conv_q_b0 = nn.Parameter(torch.zeros(self.r))
            self.conv_q_w1 = nn.Parameter(torch.empty(self.r, dh, K))
            self.conv_q_b1 = nn.Parameter(torch.zeros(self.r))
            self.conv_k_w0 = nn.Parameter(torch.empty(self.r, K))
            self.conv_k_b0 = nn.Parameter(torch.zeros(self.r))
            self.conv_k_w1 = nn.Parameter(torch.empty(self.r, dh, K))
            self.conv_k_b1 = nn.Parameter(torch.zeros(self.r))
            self.reset_conv(cfg.cca_conv_init)
        self.key_temp = nn.Parameter(torch.zeros(H))
        shift = torch.tensor([(h % 2 == 1) and self.use_v_shift for h in range(H)])
        self.register_buffer("shift_heads", shift, persistent=False)
        cos, sin = rope_tables(dh, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.collect_stats = False
        self.stats = None

    # ---- parameters -------------------------------------------------------------------
    @torch.no_grad()
    def reset_conv(self, mode: str):
        if not self.use_conv:
            return
        for w0, w1 in ((self.conv_q_w0, self.conv_q_w1), (self.conv_k_w0, self.conv_k_w1)):
            if mode == "identity":
                w0.zero_(); w0[:, -1] = 1.0
                w1.zero_()
                idx = torch.arange(self.r)
                w1[idx, idx % self.dh, -1] = 1.0
            else:
                nn.init.uniform_(w0, -1 / math.sqrt(self.K), 1 / math.sqrt(self.K))
                b = 1 / math.sqrt(self.dh * self.K)
                nn.init.uniform_(w1, -b, b)

    def new_cache(self, B, max_len, device, dtype, level=None):
        h = self.H if level is None else level
        return CCACache(B, h, max_len, self.dh, self.K, device, dtype)

    def cache_bytes_per_token(self, level=None, bytes_per_el=2):
        h = self.H if level is None else level
        return 2 * h * self.dh * bytes_per_el

    def attn_flops_per_token(self, T, level=None):
        h = self.H if level is None else level
        return 6 * T * 2 * h * self.dh

    # ---- convolutions ------------------------------------------------------------------
    def _conv(self, x, hx, hy0, w0, b0, w1, b1, heads):
        """Causal depthwise conv0 then per-head conv1 over x (B, T, c), given histories hx, hy0
        of the last K-1 conv0 inputs/outputs (zeros at the sequence start). Returns y1 and the new
        histories."""
        K, c = self.K, x.shape[-1]
        hx, hy0 = hx.to(x.dtype), hy0.to(x.dtype)
        xin = torch.cat([hx, x], dim=1)                                       # (B, K-1+T, c)
        y0 = F.conv1d(xin.transpose(1, 2), w0.unsqueeze(1).to(x.dtype), b0.to(x.dtype), groups=c)
        y0 = y0.transpose(1, 2)                                               # (B, T, c)
        y0in = torch.cat([hy0, y0], dim=1)
        y1 = F.conv1d(y0in.transpose(1, 2), w1.to(x.dtype), b1.to(x.dtype), groups=heads)
        return y1.transpose(1, 2), xin[:, -(K - 1):] if K > 1 else hx, y0in[:, -(K - 1):] if K > 1 else hy0

    # ---- forward -----------------------------------------------------------------------
    def forward(self, x, cache: Optional[CCACache] = None, start_pos=0, level=None, gates=None):
        B, T, _ = x.shape
        h = self.H if level is None else int(level)
        if cache is not None:
            assert cache.heads == h, f"cache holds {cache.heads} heads but level is {h}"
        if h == 0:
            return torch.zeros_like(x)
        c, dh = h * self.dh, self.dh
        q_pre = F.linear(x, self.q_down.weight[:c])
        k_pre = F.linear(x, self.k_down.weight[:c])
        v_all = F.linear(x, self.v_down.weight[:c])
        if self.use_conv:
            zeros = q_pre.new_zeros(B, max(self.K - 1, 0), c)
            hqx, hqy = (cache.qx, cache.qy0) if cache is not None else (zeros, zeros)
            hkx, hky = (cache.kx, cache.ky0) if cache is not None else (zeros, zeros)
            q_post, nqx, nqy = self._conv(q_pre, hqx, hqy, self.conv_q_w0[:c], self.conv_q_b0[:c],
                                          self.conv_q_w1[:c], self.conv_q_b1[:c], h)
            k_post, nkx, nky = self._conv(k_pre, hkx, hky, self.conv_k_w0[:c], self.conv_k_b0[:c],
                                          self.conv_k_w1[:c], self.conv_k_b1[:c], h)
            if cache is not None:
                cache.qx, cache.qy0, cache.kx, cache.ky0 = nqx, nqy, nkx, nky
        else:
            q_post, k_post = q_pre, k_pre
        if self.use_qk_mean:
            m = 0.5 * (q_pre + k_pre)
            q_post, k_post = q_post + m, k_post + m
        q = q_post.view(B, T, h, dh).transpose(1, 2)
        k = k_post.view(B, T, h, dh).transpose(1, 2)
        if self.collect_stats:
            self._accumulate_stats(k_post.view(B, T, h, dh), v_all.view(B, T, h, dh))
        qf, kf = q.float(), k.float()
        qf = qf * (math.sqrt(dh) / qf.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        kf = kf * (math.sqrt(dh) / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        kf = kf * torch.exp(self.key_temp[:h].float()).view(1, h, 1, 1)
        cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]
        q, k = apply_rope(qf.to(x.dtype), cos, sin), apply_rope(kf.to(x.dtype), cos, sin)
        # value shift with a fixed per-head source
        v_hist = cache.v_last.to(v_all.dtype) if cache is not None else v_all.new_zeros(B, 1, c)
        v_prev = torch.cat([v_hist, v_all[:, :-1]], dim=1)
        shift = self.shift_heads[:h].repeat_interleave(dh).view(1, 1, c)
        v = torch.where(shift, v_prev, v_all).view(B, T, h, dh).transpose(1, 2)
        if cache is not None:
            cache.v_last = v_all[:, -1:]
            k, v = cache.append(k, v)
        out = _sdpa(q, k, v, start_pos)                                       # (B, h, T, dh)
        if gates is not None:
            out = out * gates[:h].view(1, h, 1, 1).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, T, c)
        return F.linear(out, self.up.weight[:, :c])

    def _accumulate_stats(self, k_post, v_all):
        with torch.no_grad():
            ek = k_post.float().pow(2).sum(-1).sum((0, 1))
            ev = v_all.float().pow(2).sum(-1).sum((0, 1))
            n = k_post.shape[0] * k_post.shape[1]
            if self.stats is None:
                self.stats = {"k": torch.zeros(self.H, device=ek.device), "v": torch.zeros(self.H, device=ek.device), "n": 0}
            self.stats["k"][: ek.numel()] += ek
            self.stats["v"][: ev.numel()] += ev
            self.stats["n"] += n


# --------------------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d: int, ratio: int):
        super().__init__()
        self.fc1 = Linear(d, ratio * d)
        self.fc2 = Linear(ratio * d, d, residual=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def make_attention(cfg: ModelConfig) -> nn.Module:
    if cfg.attn in ("mha", "gqa"):
        return GQAttention(cfg)
    if cfg.attn == "mla":
        return MLAttention(cfg)
    if cfg.attn == "cca":
        return CCAttention(cfg)
    raise ValueError(cfg.attn)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = make_attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x, cache=None, start_pos=0, level=None, gates=None):
        x = x + self.attn(self.norm1(x), cache=cache, start_pos=start_pos, level=level, gates=gates)
        return x + self.mlp(self.norm2(x))


def _expand_levels(levels: Level, n_layers: int) -> List[Optional[int]]:
    if levels is None or isinstance(levels, int):
        return [levels] * n_layers
    levels = list(levels)
    assert len(levels) == n_layers, "per-layer levels must list every layer"
    return levels


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        std, L = self.cfg.init_std, self.cfg.n_layers
        nn.init.normal_(self.wte.weight, 0.0, std)
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.normal_(m.weight, 0.0, std / math.sqrt(2 * L) if m.residual else std)
            elif isinstance(m, RMSNorm):
                m.weight.fill_(1.0)
            elif isinstance(m, CCAttention):
                m.key_temp.zero_()
                m.reset_conv(self.cfg.cca_conv_init)
                if m.use_conv:
                    for b in (m.conv_q_b0, m.conv_q_b1, m.conv_k_b0, m.conv_k_b1):
                        b.zero_()

    # ---- bookkeeping -------------------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, seq_len: int, levels: Level = None) -> float:
        lv = _expand_levels(levels, self.cfg.n_layers)
        attn = sum(b.attn.attn_flops_per_token(seq_len, l) for b, l in zip(self.blocks, lv))
        return 6 * self.n_params() + attn

    def cache_bytes_per_token(self, levels: Level = None, bytes_per_el: int = 2) -> int:
        lv = _expand_levels(levels, self.cfg.n_layers)
        return sum(b.attn.cache_bytes_per_token(l, bytes_per_el) for b, l in zip(self.blocks, lv))

    def new_caches(self, B, max_len, device=None, dtype=None, levels: Level = None):
        device = device or self.wte.weight.device
        dtype = dtype or self.wte.weight.dtype
        lv = _expand_levels(levels, self.cfg.n_layers)
        return [b.attn.new_cache(B, max_len, device, dtype, l) for b, l in zip(self.blocks, lv)]

    def set_collect_stats(self, on: bool):
        for b in self.blocks:
            if isinstance(b.attn, CCAttention):
                b.attn.collect_stats = on
                if on:
                    b.attn.stats = None

    # ---- forward -----------------------------------------------------------------------
    def hidden(self, idx, levels: Level = None, caches=None, start_pos: int = 0, gates=None):
        x = self.wte(idx)
        lv = _expand_levels(levels, self.cfg.n_layers)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache=caches[i] if caches is not None else None, start_pos=start_pos,
                    level=lv[i], gates=gates[i] if gates is not None else None)
        return self.norm_f(x)

    def forward(self, idx, targets=None, levels: Level = None, caches=None, start_pos: int = 0, gates=None,
                multi_levels: Optional[Sequence] = None, multi_weights: Optional[Sequence[float]] = None):
        if multi_levels is not None:
            # weighted mean of the loss at several nesting levels in ONE call (true MRL objective;
            # a single forward call keeps DDP's gradient hooks simple)
            w = list(multi_weights) if multi_weights else [1.0] * len(multi_levels)
            total = sum(wi * self.forward(idx, targets, levels=lv)[1] for wi, lv in zip(w, multi_levels))
            return None, total / sum(w)
        h = self.hidden(idx, levels, caches, start_pos, gates)
        if targets is None:
            return self.lm_head(h), None
        if self.cfg.loss_chunk and self.training:
            return None, chunked_lm_loss(h, self.lm_head.weight, targets, self.cfg.loss_chunk)
        logits = self.lm_head(h)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, levels: Level = None, greedy: bool = True):
        B, T0 = idx.shape
        caches = self.new_caches(B, T0 + max_new_tokens, levels=levels)
        logits, _ = self(idx, levels=levels, caches=caches, start_pos=0)
        out = [idx]
        nxt = logits[:, -1:].argmax(-1) if greedy else torch.multinomial(logits[:, -1].softmax(-1), 1)
        pos = T0
        for _ in range(max_new_tokens - 1):
            out.append(nxt)
            logits, _ = self(nxt, levels=levels, caches=caches, start_pos=pos)
            pos += 1
            nxt = logits[:, -1:].argmax(-1) if greedy else torch.multinomial(logits[:, -1].softmax(-1), 1)
        out.append(nxt)
        return torch.cat(out, dim=1)


def _ce_chunk(h, w, t):
    logits = F.linear(h, w).float()
    return F.cross_entropy(logits, t, ignore_index=-1, reduction="sum")


def chunked_lm_loss(h, w, targets, chunk: int):
    """Mean cross-entropy without materializing all logits: each chunk's logits are recomputed in
    the backward pass (activation checkpointing). Useful for 128K-vocab models on 16 GB GPUs."""
    h2, t2 = h.reshape(-1, h.size(-1)), targets.reshape(-1)
    total = h2.new_zeros((), dtype=torch.float32)
    for s in range(0, h2.size(0), chunk):
        total = total + checkpoint(_ce_chunk, h2[s:s + chunk], w, t2[s:s + chunk], use_reentrant=False)
    return total / (t2 != -1).sum().clamp_min(1)
