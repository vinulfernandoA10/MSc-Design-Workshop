"""Run every (method, seed) combination back to back; safe to stop and rerun at any time.

Finished runs (their .npz exists) are skipped; an interrupted run resumes from its last epoch.
Each run's output is shown live and also saved to runs/log_<run name>.txt.

    python run_all.py                                  # 4 methods x seeds 0-4 on pneumoniamnist, NLL selection
    python run_all.py --data dermamnist --epochs 50
"""
import argparse, os, subprocess, sys
from train_baseline import METHODS, run_name

p = argparse.ArgumentParser()
p.add_argument("--data", default="pneumoniamnist")
p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
p.add_argument("--select", default="nll", choices=["auc", "nll"])
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--out", default="runs")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
jobs = [(m, s) for s in a.seeds for m in a.methods]  # seed-major: a full set of methods finishes early
failed = []
for i, (method, seed) in enumerate(jobs, 1):
    loss, dropout = METHODS[method]
    name = run_name(a.data, loss, seed, dropout=dropout, select=a.select)
    if os.path.exists(f"{a.out}/{name}.npz"):
        print(f"[{i}/{len(jobs)}] {name}: already done, skipping"); continue
    print(f"[{i}/{len(jobs)}] {name}: training", flush=True)
    cmd = [sys.executable, "-u", "-W", "ignore", "train_baseline.py", "--data", a.data, "--loss", loss,
           "--seed", str(seed), "--dropout", str(dropout), "--select", a.select, "--epochs", str(a.epochs), "--out", a.out]
    with open(f"{a.out}/log_{name}.txt", "a") as log, \
         subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
    if proc.returncode:
        failed.append(name); print(f"!! {name} failed (exit {proc.returncode}), continuing")

print("all done" if not failed else f"failed: {failed} - rerun this script to retry them")
