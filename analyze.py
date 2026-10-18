"""Summarise all runs of one dataset into results_<data>_<select>.md.

Table 1: clean test set, mean +- std over seeds, for each method with/without temperature scaling,
         MC Dropout, and the 5-seed ensemble.
Table 2/3: ECE and accuracy under corruption, by severity (averaged over the 5 corruption types).

    python analyze.py                          # pneumoniamnist, NLL-selected runs
    python analyze.py --select auc             # the earlier AUC-selected runs (clean table only)
"""
import argparse, glob, sys
import numpy as np
import torch
from corrupt import SEVERITY
from temp_scale import fit_temperature
from train_baseline import METHODS, metrics, run_name

NAMES = {"ce": "Cross-entropy", "ls": "Label smoothing", "focal": "Focal loss", "mcdo": "Dropout 0.5"}
SETS = ["val", "test"] + [f"c_{c}_{s}" for s in range(1, 6) for c in SEVERITY]


def probs(logits):
    return torch.softmax(torch.from_numpy(logits).float(), 1).numpy()


def score(runs, prefix="", ts=False, ensemble=False):
    """Per-run (or one ensemble) metrics on every available set: {set: metrics}."""
    vy = runs[0]["val_labels"].astype(np.int64); ty = runs[0]["test_labels"].astype(np.int64)
    sets = [s for s in SETS if f"{prefix}{s}_logits" in runs[0]]
    temps = [fit_temperature(r[f"{prefix}val_logits"], vy) if ts else 1.0 for r in runs]
    if ensemble:  # average member probabilities; log of the average is a valid logit vector for metrics()
        return [{s: metrics(np.log(np.clip(np.mean([probs(r[f"{prefix}{s}_logits"] / T) for r, T in zip(runs, temps)], 0),
                                           1e-12, None)), vy if s == "val" else ty) for s in sets}], temps
    return [{s: metrics(r[f"{prefix}{s}_logits"] / T, vy if s == "val" else ty) for s in sets} for r, T in zip(runs, temps)], temps


def by_severity(res, key):
    """Mean over runs of `key`, at severity 0 (clean) and 1..5 averaged over corruption types."""
    sev = [np.mean([r["test"][key] for r in res])]
    for s in range(1, 6):
        sev.append(np.mean([np.mean([r[f"c_{c}_{s}"][key] for c in SEVERITY]) for r in res]))
    return sev


def fmt(vals):
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


if __name__ == "__main__":  # importable: significance.py reuses score()
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pneumoniamnist")
    p.add_argument("--select", default="nll", choices=["auc", "nll"])
    p.add_argument("--runs", default="runs", help="folder with the .npz files")
    a = p.parse_args()
    # val ECE is in-sample for checkpoint selection and temperature fitting; the gap is what fails to transfer to test
    clean = ["| Method | Runs | Acc | AUC | ECE ↓ | NLL ↓ | val ECE | test − val ECE |", "|---|---|---|---|---|---|---|---|"]
    shift = {k: ["| Method | clean | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 |", "|---|---|---|---|---|---|---|"]
             for k in ["ece", "acc", "bacc"]}
    for method, (loss, dropout) in METHODS.items():
        files = sorted(glob.glob(f"{a.runs}/{run_name(a.data, loss, '[0-9]', dropout=dropout, select=a.select)}.npz"))
        if not files: continue
        runs = [np.load(f) for f in files]
        name = NAMES[method]
        variants = [(name, "", False, False), ("+ temp. scaling", "", True, False),
                    (f"{len(runs)}-seed ensemble", "", False, True)]
        if dropout:
            variants = [(f"{name} (single pass)", "", False, False), (f"MC Dropout ({name})", "mc_", False, False),
                        ("+ temp. scaling", "mc_", True, False)]
        for label, prefix, ts, ens in variants:
            res, temps = score(runs, prefix, ts, ens)
            if ts: label += f" (T = {np.mean(temps):.2f})"
            clean.append(f"| {label} | {len(runs)} | " + " | ".join(fmt([r['test'][k] for r in res]) for k in ["acc", "auc", "ece", "nll"])
                         + f" | {fmt([r['val']['ece'] for r in res])} | {fmt([r['test']['ece'] - r['val']['ece'] for r in res])} |")
            if "c_noise_1" in res[0]:
                for k in shift:
                    shift[k].append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in by_severity(res, k)) + " |")
        clean.append(f"| *best epochs: {[int(r['best_epoch']) for r in runs]}* | | | | | | | |")

    out = [f"# Results: {a.data} (28px, checkpoint selected by val {a.select.upper()})", "",
           "## Clean test set (mean ± std over seeds)", ""] + clean
    if len(shift["ece"]) > 2:
        out += ["", "## ECE ↓ under corruption (mean over seeds and the 5 corruption types)", ""] + shift["ece"]
        out += ["", "## Accuracy under corruption", ""] + shift["acc"]
        out += ["", "## Balanced accuracy under corruption (mean per-class recall; immune to majority-class collapse)", ""] + shift["bacc"]
    out = "\n".join(out)
    dest = f"results_{a.data}_{a.select}.md" if a.runs == "runs" else f"{a.runs}/results_{a.data}_{a.select}.md"
    open(dest, "w", encoding="utf-8").write(out + "\n")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which lacks ± and ↓
    print(out)
