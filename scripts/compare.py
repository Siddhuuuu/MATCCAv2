"""
compare.py - aggregate runs/*/results.json across seeds and print paired comparisons.

  python scripts/compare.py --runs runs --baseline s_mha --seeds-from-name

Runs of the same config with different seeds must be named <config>_seed<k> (use
--set run_name=s_cca4_seed1 seed=1 data_seed=1 when launching). For each config and level, prints
mean, 95% t-interval and paired p-value against the baseline at the same seeds.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from matcca.stats import paired_test, t_interval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()
    table = defaultdict(dict)  # (config, level) -> {seed: loss}
    for p in glob.glob(os.path.join(args.runs, "*", "results.json")):
        r = json.load(open(p))
        m = re.match(r"(.*)_seed(\d+)$", r["run"])
        name, seed = (m.group(1), int(m.group(2))) if m else (r["run"], 0)
        for lvl, loss in r["final_val"].items():
            table[(name, lvl)][seed] = loss
    base = {s: v for (n, l), d in table.items() if n == args.baseline for s, v in d.items()}
    print(f"{'config':26s} {'level':>6s} {'n':>2s} {'mean':>8s} {'95% CI':>19s} {'vs base':>9s} {'p(paired)':>9s}")
    for (name, lvl), d in sorted(table.items()):
        seeds = sorted(d)
        ci = t_interval([d[s] for s in seeds])
        common = [s for s in seeds if s in base]
        cmp = paired_test([d[s] for s in common], [base[s] for s in common]) if len(common) >= 2 else {}
        print(f"{name:26s} {lvl:>6s} {ci['n']:2d} {ci['mean']:8.4f} [{ci['lo']:8.4f},{ci['hi']:8.4f}] "
              f"{cmp.get('diff', float('nan')):+9.4f} {cmp.get('p_paired', float('nan')):9.3f}")


if __name__ == "__main__":
    main()
