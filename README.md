# MatCCA v2

Elastic compressed convolutional attention (MatCCA) and its baselines, rebuilt so that every number
in the paper can be reproduced from one config file on a Kaggle T4.

v1 (the MLForSys submission) produced its tables from `%%writefile` cells inside `MatCCA.ipynb`;
the `.py` files on GitHub were older. This package replaces both. `audit/` reproduces the problems
found in v1 so you can check them yourself.

## What changed from v1

| v1 problem | Effect on the paper | v2 | Guarded by |
| --- | --- | --- | --- |
| Tied embedding at PyTorch's N(0, 1) default | step-0 loss 330 nats (ln 8000 = 8.99) | N(0, 0.02) init, residual projections / sqrt(2L) | `test_init_loss_close_to_uniform` |
| Value latents laid out `[v_now \| v_prev]` | lower ranks are not sub-networks; heads change value source when truncated | value source fixed per head (even: current, odd: previous token) | `test_lower_level_is_exact_subnetwork` |
| No KV cache or decode path | serving claims unmeasured | real caches for MHA/GQA/MLA/CCA, prefill, chunked prefill, decode, in-place head truncation | `test_decode_matches_full_forward`, `test_inflight_cache_truncation_single_layer` |
| Allocations rounded after the fact, floor of one head | WF/OWF scored with 12-50% more cache than greedy | every allocator picks heads per layer under an exact budget, same feasible set | `test_run_all_exact_budgets`, `test_dp_is_exact` |
| Allocators scored on their calibration batches | in-sample numbers | calibration from training shards, scoring on fixed validation windows | `scripts/run_allocators.py` |
| Random validation windows per call | +/-0.02 nats noise between evaluations | full, fixed, non-overlapping validation windows | `test_data_is_deterministic_and_paired` |
| Data order tied to model init RNG | no paired comparisons | `data_seed` controls data order, identical across architectures | same |
| Unseeded level sampling, resume without RNG state | runs not reproducible | every RNG checkpointed; resume is bit-exact on CPU | `test_resume_is_exact` |
| fp32, constant LR, no clipping, 5k steps | ~4% of T4 peak; unconverged comparisons | fp16 autocast + GradScaler, compile, fused AdamW, warmup + cosine, clipping, token budgets | `test_training_reduces_loss` |
| Learned absolute positions | MHA had no relative-position bias, CCA did (convs) | RoPE in every attention type | - |
| MLA without latent norm | under-tuned baseline | RMSNorm on the KV latent + decoupled RoPE key (DeepSeek-V2) | `test_decode_matches_full_forward[mla]` |
| Dedicated CCA 8x/16x narrowed heads, MatCCA dropped heads | premium compared different architectures | both geometries available (`cca8`, `cca8_narrow`) | configs |

## Layout

```
matcca/model.py      GPT backbone; MHA/GQA, MLA, CCA with exact nesting; caches; chunked LM loss
matcca/data.py       uint16 token shards (modded-nanogpt format), deterministic train/val windows
matcca/train.py      training run from a JSON config (1 GPU or torchrun DDP), exact resume
matcca/evaluate.py   validation loss, in-flight downshift quality, decode benchmark
matcca/allocate.py   uniform, random, variance, Fisher+DP, profile+DP, greedy-KL under exact budgets
matcca/stats.py      t-intervals, paired/Welch tests, bootstrap
scripts/             data download, config generation, decode benchmark, downshift, allocators, comparison
configs/s, configs/m experiment matrix at 51M and 124M parameters
audit/               scripts that reproduce the v1 findings from the notebook
tests/               37 CPU tests (python -m pytest -q tests)
```

## Quick start (any machine)

```bash
pip install -r requirements.txt
python -m pytest -q tests                       # ~40 s on CPU
python scripts/make_configs.py                  # (configs are already generated)
python scripts/prepare_fineweb.py --train-shards 10
python -m matcca.train --config configs/s/mha.json --set max_steps=50 run_name=calib eval_every=1000000
```

The last command prints `tok_s` and `tflops`. Every compute estimate in the plan assumes 15 TFLOP/s per
T4; rescale by your measured number.

## Experiment matrix

| Config | What it answers |
| --- | --- |
| `mha`, `gqa`, `mla` | baselines; `gqa` has the cache of `cca4`, `mla` the cache of `cca8` |
| `cca1x` | CCA without compression: do the convolutions, not compression, beat MHA? |
| `cca4`, `cca8`, `cca16` | dedicated compressed models (head-dropping geometry, like MatCCA) |
| `cca8_narrow` | v1's dedicated geometry (narrower heads) |
| `cca4_noconv`, `cca4_noqkmean`, `cca4_novshift` | component ablations |
| `matcca_strat`, `matcca_random`, `matcca_sum` | nesting at equal cost per step (strat, random) and the true MRL objective (sum, 3x cost) |

Seeds: launch with `--set seed=k data_seed=k run_name=<cfg>_seed<k>`; the same `data_seed` gives every
architecture identical batches, so `scripts/compare.py` can use paired tests.

## After training

```bash
python scripts/compare.py --runs runs --baseline s_mha
python scripts/downshift_eval.py --ckpt runs/s_matcca_strat_seed0/ckpt.pt --low 4 2 --splits 256 512 768
python scripts/run_allocators.py --ckpt runs/s_matcca_strat_seed0/ckpt.pt --budgets 48 32 24 16 12
python scripts/bench_decode.py --configs configs/s/mha.json configs/s/gqa.json configs/s/mla.json \
    configs/s/cca4.json configs/s/matcca_strat.json --batches 1 8 32 128 256 --prompt 1024 --gen 128
```

## Kaggle

See `kaggle/README.md` for the cell-by-cell recipe (T4 or T4 x2, Hugging Face Hub checkpoints,
resuming across sessions and accounts).

## Not done yet

* Retrofit of pretrained models (Track B: SmolLM2-360M, Llama-3.2-1B). The model code supports
  per-head levels and caches; the conversion script (RoPE-pair selection for keys, PCA for values,
  identity-initialized convolutions, attention-only KL distillation) is the next milestone.
* Eviction (StreamingLLM/H2O/SnapKV) and 2-bit KV quantization baselines for the downshift experiment.
* A Triton decode kernel for latent attention with per-request head counts, and the serving simulator.
* MLA decode uses the non-absorbed form, so its decode speed is pessimistic.
