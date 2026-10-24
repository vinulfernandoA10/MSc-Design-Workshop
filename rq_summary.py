"""Cross-dataset answers to RQ3 and RQ4 -> rq3_rq4_summary.md

RQ3  Does better calibration mean better selective prediction? Spearman correlation between ECE and
     E-AURC across all single-model runs (every method, with/without temperature scaling), per dataset.
RQ4  Do cheap methods approach deep ensembles? Cost vs benefit averaged over datasets, plus paired
     Wilcoxon tests (Holm) of each 1x-cost method against the cross-entropy ensemble, paired by
     (dataset, seed). The ensemble value repeats within a dataset, so these pairs are not fully
     independent: read the p-values as indicative.

    python rq_summary.py --data pneumoniamnist dermamnist bloodmnist organcmnist
"""
import argparse, glob, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from analyze import score
from corrupt import SEVERITY
from significance import VARIANTS, collect, holm
from train_baseline import METHODS, run_name

SEV5 = lambda r, k: np.mean([r[f"c_{c}_5"][k] for c in SEVERITY])
COLS = {  # column -> extractor, lower is better
    "ECE clean": (lambda r: r["test"]["ece"], True), "ECE sev 5": (lambda r: SEV5(r, "ece"), True),
    "E-AURC clean": (lambda r: r["test"]["eaurc"], True), "E-AURC sev 5": (lambda r: SEV5(r, "eaurc"), True),
    "Bal. acc sev 5": (lambda r: SEV5(r, "bacc"), False)}
COST = {  # row -> (base variant or ensemble method, training cost, test-time forward passes)
    "Cross-entropy": ("Cross-entropy", 1, 1), "CE + temp. scaling": ("CE + temp. scaling", 1, 1),
    "Label smoothing": ("Label smoothing", 1, 1), "Focal loss": ("Focal loss", 1, 1),
    "MC Dropout": ("MC Dropout", 1, 20), "CE ensemble (5)": ("ens:ce", 5, 5),
    "LS ensemble (5)": ("ens:ls", 5, 5), "Focal ensemble (5)": ("ens:focal", 5, 5)}


def ensembles(datasets, select):
    """{method: {dataset: ensemble result dict}}"""
    out = {}
    for m in ("ce", "ls", "focal"):
        loss, dropout = METHODS[m]
        for d in datasets:
            files = sorted(glob.glob(f"runs/{run_name(d, loss, '[0-9]', dropout=dropout, select=select)}.npz"))
            if files: out.setdefault(m, {})[d] = score([np.load(f) for f in files], ensemble=True)[0][0]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["pneumoniamnist", "dermamnist", "bloodmnist", "organcmnist"])
    p.add_argument("--select", default="nll", choices=["auc", "nll"])
    a = p.parse_args()
    res, ens = collect(a.data, a.select, "runs"), ensembles(a.data, a.select)
    L = [f"# RQ3 and RQ4 summary ({', '.join(a.data)}; checkpoint selected by val {a.select.upper()})", ""]

    # RQ3: rank correlation between calibration error and selective-prediction quality
    L += ["## RQ3: is a better-calibrated model better at selective prediction?", "",
          "Spearman ρ between ECE and E-AURC over all single-model runs (4 methods × ± temperature scaling × 5 seeds).",
          "ρ near 0 = calibration and the ability to rank right above wrong are unrelated.", "",
          "| Dataset | runs | ρ clean (p) | ρ severity 5 (p) |", "|---|---|---|---|"]
    for d in a.data:
        rows = [r for v in VARIANTS for (dd, _), r in res[v].items() if dd == d]
        if not rows: continue
        cells = []
        for get_e, get_s in [(lambda r: r["test"]["ece"], lambda r: r["test"]["eaurc"]),
                             (lambda r: SEV5(r, "ece"), lambda r: SEV5(r, "eaurc"))]:
            rho, pv = spearmanr([get_e(r) for r in rows], [get_s(r) for r in rows])
            cells.append(f"{rho:+.2f} ({pv:.3f})")
        L.append(f"| {d} | {len(rows)} | " + " | ".join(cells) + " |")

    # RQ4: cost vs benefit, each dataset averaged over seeds first, then over datasets
    L += ["", "## RQ4: cost vs benefit (mean over datasets of the per-dataset seed mean)", "",
          "| Method | training cost | test passes | " + " | ".join(COLS) + " |", "|---|---|---|" + "---|" * len(COLS)]
    for row, (src, train, test) in COST.items():
        vals = []
        for col, (get, _) in COLS.items():
            if src.startswith("ens:"):
                per_d = [get(r) for r in ens.get(src[4:], {}).values()]
            else:
                per_d = [np.mean([get(r) for (dd, _), r in res[src].items() if dd == d]) for d in a.data
                         if any(dd == d for dd, _ in res[src])]
            vals.append(f"{np.mean(per_d):.3f}" if per_d else "–")
        L.append(f"| {row} | {train}× | {test}× | " + " | ".join(vals) + " |")

    # RQ4 tests: single 1x-cost models vs the cross-entropy ensemble, paired by (dataset, seed)
    L += ["", "## RQ4: is a single model as good as the cross-entropy ensemble?", "",
          "Paired by (dataset, seed) against that dataset's CE ensemble; two-sided Wilcoxon, Holm within each metric.", ""]
    for col in ["ECE sev 5", "E-AURC sev 5"]:
        get, lower = COLS[col]
        rows, ps = [], []
        for v in ["Cross-entropy", "CE + temp. scaling", "Label smoothing", "Focal loss", "MC Dropout"]:
            keys = [k for k in sorted(res[v]) if k[0] in ens.get("ce", {})]
            diff = np.array([get(res[v][k]) - get(ens["ce"][k[0]]) for k in keys])
            wins = int(np.sum(diff < 0) if lower else np.sum(diff > 0))
            rows.append((v, len(keys), np.median(diff), wins)); ps.append(wilcoxon(diff).pvalue)
        L += [f"### {col} ({'lower' if lower else 'higher'} is better)", "",
              "| Single model | vs | n | Δ (single − ensemble) | single wins | p (Holm) |", "|---|---|---|---|---|---|"]
        for (v, n, d, w), pa in zip(rows, holm(np.array(ps))):
            L.append(f"| {v} | CE ensemble | {n} | {d:+.3f} | {w}/{n} | {pa:.4f}{' *' if pa < .05 else ''} |")
        L.append("")

    out = "\n".join(L)
    open("rq3_rq4_summary.md", "w", encoding="utf-8").write(out + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
