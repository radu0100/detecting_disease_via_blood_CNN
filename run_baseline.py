"""Run the ResNet-18 baseline: 2 tasks x 3 seeds. Skips runs that already finished.

    python run_baseline.py
    python run_baseline.py --tasks cnmc --seeds 0     # quick single run
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", default=["pbc", "cnmc"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--model", default="resnet18")
args, extra = ap.parse_known_args()

for task in args.tasks:
    for seed in args.seeds:
        done = ROOT / "runs" / f"{task}_{args.model}_baseline_s{seed}" / "metrics.json"
        if done.exists() and not extra:
            print(f"skip {done.parent.name} (done)")
            continue
        cmd = [sys.executable, str(ROOT / "train.py"), "--task", task, "--model", args.model,
               "--seed", str(seed), *extra]
        subprocess.run(cmd, check=True, cwd=ROOT)
