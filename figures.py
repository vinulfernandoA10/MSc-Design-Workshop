"""Thesis figures from the saved logits (no retraining). Writes figures/*.png (300 dpi) and *.pdf.

    python figures.py

Fig 1  ECE vs corruption severity, per dataset (RQ1)
Fig 2  Change in ECE from temperature scaling vs severity, per dataset (RQ2)
Fig 3  Reliability diagrams, clean vs severity 5, cross-entropy vs focal loss. ECE here is computed on the
       pooled predictions, so over- and underconfidence on different corruptions can cancel: it is lower
       than the per-corruption mean ECE in the results tables.

Colours: validated categorical palette (adjacent pairs CVD-safe). Aqua and yellow sit below 3:1
contrast on the surface, so every series also has its own marker and a direct label.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import by_severity, probs, score
from corrupt import SEVERITY
from train_baseline import METHODS, ece, run_name

DATASETS = {"pneumoniamnist": "PneumoniaMNIST", "dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST"}
SERIES = [  # method, logit prefix, label, colour (fixed slot order), marker
    ("ce", "", "Cross-entropy", "#2a78d6", "o"), ("ls", "", "Label smoothing", "#eb6834", "s"),
    ("focal", "", "Focal loss", "#1baf7a", "^"), ("mcdo", "mc_", "MC Dropout", "#eda100", "D")]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
X, XT = np.arange(6), ["clean", "1", "2", "3", "4", "5"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "text.color": INK})


def runs(data, method):
    loss, dropout = METHODS[method]
    return [np.load(f) for f in sorted(glob.glob(f"runs/{run_name(data, loss, '[0-9]', dropout=dropout, select='nll')}.npz"))]


def end_labels(ax, ends):
    """Direct labels at the right end of each line, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim(); gap = 0.075 * (hi - lo); placed = []
    for y, text in sorted(ends):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.text(5.25, y, text, color=INK2, fontsize=8, va="center")
    ax.set_xlim(-0.3, 7.6)


def severity_figure(metric_fn, ylabel, title, fname, zero_line=False):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (data, dname) in zip(axes, DATASETS.items()):
        ends = []
        for method, prefix, label, colour, marker in SERIES:
            y = metric_fn(runs(data, method), prefix)
            ax.plot(X, y, color=colour, lw=2, marker=marker, ms=5, mec=SURF, mew=1, label=label, zorder=3)
            ends.append((y[-1], label))
        if zero_line: ax.axhline(0, color=BASE, lw=1, zorder=1)
        ax.set_title(dname, loc="left"); ax.set_xticks(X, XT); ax.set_xlabel("Corruption severity")
        ax.set_xlim(-0.3, 5.3)
    end_labels(axes[-1], ends)
    axes[0].set_ylabel(ylabel)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, x=0.01, y=1.1, ha="left", fontsize=11, color=INK)
    save(fig, fname)


def save(fig, fname):
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/{fname}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"figures/{fname}.png/.pdf")


def ece_curve(rs, prefix):
    return by_severity(score(rs, prefix)[0], "ece")


def ts_delta(rs, prefix):
    return np.array(by_severity(score(rs, prefix, ts=True)[0], "ece")) - np.array(ece_curve(rs, prefix))


def reliability(rs, keys, n_bins=10):
    """Pooled over seeds (and corruption types): mean confidence and accuracy per confidence bin."""
    p = np.concatenate([probs(r[k]) for r in rs for k in keys])
    y = np.concatenate([r["val_labels" if k.startswith("val") else "test_labels"] for r in rs for k in keys])
    conf, correct = p.max(1), p.argmax(1) == y
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    keep = [i for i in range(n_bins) if (b == i).sum() >= 20]  # drop near-empty bins: their accuracy is noise
    return [conf[b == i].mean() for i in keep], [correct[b == i].mean() for i in keep], ece(p, y)


def reliability_figure():
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
    for col, (data, dname) in enumerate(DATASETS.items()):
        for row, (keys, cond) in enumerate([(["test_logits"], "clean"),
                                            ([f"c_{c}_5_logits" for c in SEVERITY], "severity 5")]):
            ax = axes[row, col]; ax.grid(axis="both")
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
            notes = []
            for method, prefix, label, colour, marker in [SERIES[0], SERIES[2]]:
                x, y, e = reliability(runs(data, method), keys)
                ax.plot(x, y, color=colour, lw=2, marker=marker, ms=5, mec=SURF, mew=1, label=label, zorder=3)
                notes.append(f"{label}: pooled ECE {e:.3f}")
            ax.text(0.03, 0.97, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=8, color=INK2)
            ax.set_title(f"{dname}, {cond}", loc="left")
            ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
            if row == 1: ax.set_xlabel("Confidence")
            if col == 0: ax.set_ylabel("Accuracy")
    axes[0, 0].text(0.16, 0.24, "perfect calibration", color=MUTED, fontsize=8, rotation=45, rotation_mode="anchor")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Reliability diagrams (seeds and corruption types pooled): below the diagonal = overconfident",
                 x=0.01, y=1.04, ha="left", fontsize=11, color=INK)
    save(fig, "fig3_reliability")


if __name__ == "__main__":
    severity_figure(ece_curve, "ECE (lower is better)",
                    "Calibration error grows with corruption severity (mean of 5 seeds)", "fig1_ece_vs_severity")
    severity_figure(ts_delta, "Change in ECE from temperature scaling\n(below 0 = helps, above 0 = hurts)",
                    "Effect of temperature scaling on calibration under corruption", "fig2_temp_scaling_effect",
                    zero_line=True)
    reliability_figure()
