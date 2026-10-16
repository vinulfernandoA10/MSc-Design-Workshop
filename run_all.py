"""Run every (loss, seed) combination back to back; safe to stop and rerun at any time.

Finished runs (their .npz exists) are skipped; an interrupted run resumes from its last epoch.
Each run's output is shown live and also saved to runs/log_<data>_<loss>_s<seed>.txt.

    python run_all.py                      # 3 losses x seeds 0-4 on pneumoniamnist
    python run_all.py --data dermamnist
"""
import argparse, os, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument("--data", default="pneumoniamnist")
p.add_argument("--losses", nargs="+", default=["ce", "ls", "focal"])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
a = p.parse_args()

os.makedirs("runs", exist_ok=True)
jobs = [(l, s) for s in a.seeds for l in a.losses]  # seed-major: a full set of methods finishes early
failed = []
for i, (loss, seed) in enumerate(jobs, 1):
    name = f"{a.data}_28_{loss}_s{seed}"
    if os.path.exists(f"runs/{name}.npz"):
        print(f"[{i}/{len(jobs)}] {name}: already done, skipping"); continue
    print(f"[{i}/{len(jobs)}] {name}: training", flush=True)
    cmd = [sys.executable, "-u", "-W", "ignore", "train_baseline.py",
           "--data", a.data, "--loss", loss, "--seed", str(seed)]
    with open(f"runs/log_{name}.txt", "a") as log, \
         subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
    if proc.returncode:
        failed.append(name); print(f"!! {name} failed (exit {proc.returncode}), continuing")

print("all done" if not failed else f"failed: {failed} - rerun this script to retry them")
