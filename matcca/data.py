"""
data.py - token shards and deterministic batching.

Shard format (same as modded-nanogpt / llm.c, so kjj0/fineweb10B-gpt2 shards work directly):
  256 int32 header [magic=20240520, version=1, n_tokens, 0...] followed by n_tokens uint16 tokens.

Training batches are a pure function of (data_seed, global window offset): every architecture
trained with the same data_seed sees exactly the same windows in the same order (paired seeds),
and a resumed run continues with exactly the batches it would have seen.
Validation is a fixed, non-overlapping, sequential set of windows.
"""
from __future__ import annotations

import glob
import os
from typing import Iterator, List, Tuple

import numpy as np
import torch

MAGIC, VERSION, HEADER_INTS = 20240520, 1, 256


def write_shard(path: str, tokens) -> None:
    tokens = np.asarray(tokens)
    assert tokens.max() < 2 ** 16, "uint16 shards need vocab < 65536"
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(tokens)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def read_shard(path: str) -> np.memmap:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC and header[1] == VERSION, f"{path}: not a token shard"
    n = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(n,))


def expand(pattern_or_list) -> List[str]:
    if isinstance(pattern_or_list, (list, tuple)):
        out = []
        for p in pattern_or_list:
            out.extend(sorted(glob.glob(p)))
        return out
    return sorted(glob.glob(pattern_or_list))


class TrainWindows:
    """Non-overlapping windows of seq_len+1 tokens (stride seq_len) over all shards, visited in a
    seeded random order that is re-drawn each epoch."""

    def __init__(self, paths: List[str], seq_len: int, data_seed: int):
        assert paths, "no training shards found"
        self.shards = [read_shard(p) for p in paths]
        self.seq_len, self.seed = seq_len, data_seed
        idx = []
        for s, arr in enumerate(self.shards):
            n = (len(arr) - 1) // seq_len
            idx.append(np.stack([np.full(n, s, dtype=np.int64), np.arange(n, dtype=np.int64) * seq_len], 1))
        self.index = np.concatenate(idx, 0)
        self.n = len(self.index)
        self._perm_epoch, self._perm = None, None
        assert self.n > 0, "shards too small for this sequence length"

    @property
    def n_tokens(self) -> int:
        return self.n * self.seq_len

    def _permutation(self, epoch: int) -> np.ndarray:
        if self._perm_epoch != epoch:
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.n)
            self._perm_epoch = epoch
        return self._perm

    def batch(self, offset: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        L = self.seq_len
        out = np.empty((batch_size, L + 1), dtype=np.int64)
        for b in range(batch_size):
            epoch, j = divmod(offset + b, self.n)
            s, o = self.index[self._permutation(epoch)[j]]
            out[b] = self.shards[s][o:o + L + 1]
        t = torch.from_numpy(out)
        return t[:, :-1], t[:, 1:]


class ValWindows:
    """The first max_tokens of a held-out shard, as sequential non-overlapping windows."""

    def __init__(self, path: str, seq_len: int, max_tokens: int):
        arr = read_shard(path)
        n = min((len(arr) - 1) // seq_len, max(1, max_tokens // seq_len))
        self.arr, self.seq_len, self.n = arr, seq_len, n

    def batches(self, batch_size: int, rank: int = 0, world: int = 1) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        L = self.seq_len
        ids = list(range(rank, self.n, world))
        for i in range(0, len(ids), batch_size):
            rows = [np.asarray(self.arr[w * L: w * L + L + 1], dtype=np.int64) for w in ids[i:i + batch_size]]
            t = torch.from_numpy(np.stack(rows))
            yield t[:, :-1], t[:, 1:]
