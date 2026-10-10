# Running MatCCA v2 on Kaggle

One notebook per run. Settings: Accelerator = GPU T4 x2 (or GPU T4), Internet = On.
Add a Kaggle Secret named `HF_TOKEN` (a Hugging Face write token) and create one private model repo
on the Hub, for example `your-hf-name/matcca-runs`, shared with your collaborator.

## Cells

```python
# 1. code
!git clone https://github.com/your-github-name/matcca-v2.git
%cd matcca-v2
!pip install -q -r requirements.txt
```

```python
# 2. Hugging Face token (checkpoints survive the session and move between accounts)
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```python
# 3. data: 10 shards = 1.0B tokens for an S run, 25 for an M run
!python scripts/prepare_fineweb.py --train-shards 10
```

```python
# 4. first session only: tests and a throughput check
!python -m pytest -q tests
!python -m matcca.train --config configs/s/mha.json --set max_steps=60 run_name=calib eval_every=1000000 log_every=10
```

```python
# 5. the run (T4 x2). time_limit_min=660 checkpoints and exits before Kaggle's 12-hour limit.
HUB = "your-hf-name/matcca-runs"      # your private Hub repo
CFG, SEED = "configs/s/cca4.json", 0
RUN = f"s_cca4_seed{SEED}"
!torchrun --nproc_per_node=2 -m matcca.train --config {CFG} \
    --set run_name={RUN} seed={SEED} data_seed={SEED} hub_repo={HUB}
# single T4:  !python -m matcca.train --config {CFG} --set run_name={RUN} seed={SEED} data_seed={SEED} hub_repo={HUB}
```

Commit with **Save Version -> Save & Run All** so the run continues with the browser closed. To
continue a paused run, from any account, re-run the same cells: the trainer pulls
`<RUN>/ckpt.pt` from the Hub and resumes exactly where it stopped. `global_batch_tokens` keeps the
optimizer batch identical on one or two GPUs.

## Rules that keep the study clean

* One owner per run. Keep a shared run list (a sheet, or `runs.md` in the Hub repo) with run name,
  owner account and status before starting anything.
* Never change a config after its first run has started; make a new config instead.
* Launch every architecture of a seed with the same `data_seed`.
* If `torch.compile` fails or is slow on your Kaggle image, add `compile=false`; it changes speed,
  not results.
* Check the weekly quota counter after a short T4 x2 session to see how fast it draws quota.
