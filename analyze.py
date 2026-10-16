"""Summarise all runs: mean +- std over seeds, with/without temperature scaling, plus 5-seed deep ensembles.

    python analyze.py                  # pneumoniamnist, writes results_<data>.md
"""
import argparse, glob, sys
import numpy as np
import torch
from train_baseline import metrics
from temp_scale import fit_temperature

p = argparse.ArgumentParser()
p.add_argument("--data", default="pneumoniamnist")
a = p.parse_args()
KEYS = ["acc", "auc", "ece", "nll"]
NAMES = {"ce": "Cross-entropy", "ls": "Label smoothing", "focal": "Focal loss"}


def row(label, ms, n=None):
    cells = [f"{np.mean([m[k] for m in ms]):.3f} ± {np.std([m[k] for m in ms]):.3f}" if len(ms) > 1
             else f"{ms[0][k]:.3f}" for k in KEYS]
    return f"| {label} | {n or len(ms)} | " + " | ".join(cells) + " |"


lines = [f"# Results: {a.data} (28px, best val-AUC checkpoint)", "",
         "| Method | Runs | Acc | AUC | ECE ↓ | NLL ↓ |", "|---|---|---|---|---|---|"]
for loss, name in NAMES.items():
    runs = [np.load(f) for f in sorted(glob.glob(f"runs/{a.data}_28_{loss}_s[0-9].npz"))]
    if not runs: continue
    ty = runs[0]["test_labels"].astype(np.int64)
    vy = runs[0]["val_labels"].astype(np.int64)
    plain, scaled, temps = [], [], []
    for r in runs:
        T = fit_temperature(r["val_logits"], vy); temps.append(T)
        plain.append(metrics(r["test_logits"], ty)); scaled.append(metrics(r["test_logits"] / T, ty))
    # ensemble: average member probabilities, then pass log-probs through metrics() (softmax of log p = p)
    probs = np.mean([torch.softmax(torch.from_numpy(r["test_logits"]), 1).numpy() for r in runs], 0)
    ens = metrics(np.log(np.clip(probs, 1e-12, None)), ty)
    epochs = [int(r["best_epoch"]) for r in runs]
    lines += [row(name, plain),
              row(f"+ temp. scaling (T = {np.mean(temps):.2f} ± {np.std(temps):.2f})", scaled),
              row(f"{len(runs)}-seed ensemble", [ens], n=len(runs)),
              f"| *best epochs: {epochs}* | | | | | |"]

out = "\n".join(lines)
open(f"results_{a.data}.md", "w", encoding="utf-8").write(out + "\n")
sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which lacks ± and ↓
print(out)
