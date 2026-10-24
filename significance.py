"""Paired significance tests: does each method differ from its reference, pooled over datasets and seeds?

Unit of comparison = one (dataset, seed) pair. Runs with the same seed share weight initialisation
and data order, so pairing by seed removes that source of variance. Test: two-sided Wilcoxon
signed-rank, Holm-corrected for multiple comparisons within each metric. Ensembles are excluded
(only one per dataset, so too few to test).

    python significance.py --data pneumoniamnist dermamnist bloodmnist
"""
import argparse, os, sys
import numpy as np
from scipy.stats import wilcoxon
from analyze import score
from corrupt import SEVERITY
from train_baseline import METHODS, run_name

VARIANTS = {  # name -> (method, logit prefix, temperature scaling)
    "Cross-entropy": ("ce", "", False), "CE + temp. scaling": ("ce", "", True),
    "Label smoothing": ("ls", "", False), "LS + temp. scaling": ("ls", "", True),
    "Focal loss": ("focal", "", False), "Focal + temp. scaling": ("focal", "", True),
    "MC Dropout": ("mcdo", "mc_", False), "MC Dropout + temp. scaling": ("mcdo", "mc_", True),
}
COMPARISONS = [  # (A, B): is A different from B?
    ("Label smoothing", "Cross-entropy"), ("Focal loss", "Cross-entropy"), ("MC Dropout", "Cross-entropy"),
    ("CE + temp. scaling", "Cross-entropy"), ("LS + temp. scaling", "Label smoothing"),
    ("Focal + temp. scaling", "Focal loss"), ("MC Dropout + temp. scaling", "MC Dropout"),
]
METRICS = {  # name -> (extractor, lower is better)
    "ECE, clean test": (lambda r: r["test"]["ece"], True),
    "ECE, severity 5": (lambda r: np.mean([r[f"c_{c}_5"]["ece"] for c in SEVERITY]), True),
    "AUC, clean test": (lambda r: r["test"]["auc"], False),
    "E-AURC, clean test (RQ3)": (lambda r: r["test"]["eaurc"], True),
    "E-AURC, severity 5 (RQ3)": (lambda r: np.mean([r[f"c_{c}_5"]["eaurc"] for c in SEVERITY]), True),
}


def holm(ps):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    order, m, adj, running = np.argsort(ps), len(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i])); adj[i] = running
    return adj


def collect(datasets, select, runs_dir):
    """{variant: {(dataset, seed): per-run result dict}} for every run that exists."""
    out = {v: {} for v in VARIANTS}
    for d in datasets:
        for v, (method, prefix, ts) in VARIANTS.items():
            loss, dropout = METHODS[method]
            for seed in range(10):
                f = f"{runs_dir}/{run_name(d, loss, seed, dropout=dropout, select=select)}.npz"
                if os.path.exists(f):
                    out[v][(d, seed)] = score([np.load(f)], prefix, ts)[0][0]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["pneumoniamnist"])
    p.add_argument("--select", default="nll", choices=["auc", "nll"])
    p.add_argument("--runs", default="runs")
    a = p.parse_args()
    res = collect(a.data, a.select, a.runs)

    lines = [f"# Significance tests ({', '.join(a.data)}; checkpoint selected by val {a.select.upper()})", "",
             "Paired by (dataset, seed); two-sided Wilcoxon signed-rank; Holm-corrected within each metric.",
             "*Wins* = pairs where A is better than B. Δ = median of (A − B).", ""]
    for metric, (get, lower) in METRICS.items():
        rows, ps = [], []
        for A, B in COMPARISONS:
            keys = sorted(set(res[A]) & set(res[B]))
            if not keys or ("severity 5" in metric and "c_noise_5" not in res[A][keys[0]]): continue
            diff = np.array([get(res[A][k]) - get(res[B][k]) for k in keys])
            diff[np.abs(diff) < 1e-9] = 0  # float noise is a tie (e.g. temperature scaling cannot change AUC)
            wins = int(np.sum(diff < 0) if lower else np.sum(diff > 0))
            pval = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            rows.append((A, B, len(keys), np.median(diff), wins)); ps.append(pval)
        if not rows: continue
        adj = holm(np.array(ps))
        lines += [f"## {metric} ({'lower' if lower else 'higher'} is better)", "",
                  "| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
        for (A, B, n, d, w), pv, pa in zip(rows, ps, adj):
            lines.append(f"| {A} | {B} | {n} | {d:+.3f} | {w}/{n} | {pv:.4f} | {pa:.4f}{' *' if pa < .05 else ''} |")
        lines.append("")
    n_min = min((len(set(res[A]) & set(res[B])) for A, B in COMPARISONS), default=0)
    if n_min < 6:
        lines.append(f"> With n = {n_min} pairs, a two-sided Wilcoxon test cannot reach p < 0.05 "
                     "(smallest possible p = 0.0625 at n = 5). Pool more datasets for significance.")
    out = "\n".join(lines)
    open(f"significance_{'_'.join(a.data)}_{a.select}.md", "w", encoding="utf-8").write(out + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(out)


if __name__ == "__main__":
    # self-check for the Holm correction: classic textbook example
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    main()
