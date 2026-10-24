"""Out-of-distribution (OOD) detection with the saved models -> results_ood.md

Each model sees 2,000 test images from each *other* dataset (channels converted to match) and should
be less confident on them. Score = maximum softmax probability (Hendrycks & Gimpel, 2017); metric =
AUROC for separating the model's own test set (in-distribution) from each OOD set (0.5 = chance).
Inference only, cached per model in runs/ood_<run>.npz, so an interrupted run resumes.

    python ood.py
"""
import glob, os, sys
import numpy as np
import torch
import medmnist
from sklearn.metrics import roc_auc_score
from analyze import probs
from significance import holm
from scipy.stats import wilcoxon
from train_baseline import METHODS, predict_x, resnet18, run_name

DATASETS = ["pneumoniamnist", "dermamnist", "bloodmnist", "organcmnist"]
N_OOD, MC = 2000, 20
NAMES = {"ce": "Cross-entropy", "ls": "Label smoothing", "focal": "Focal loss", "mcdo": "Dropout 0.5"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def images(data, in_ch):
    """First N_OOD test images of `data`, as normalised tensors with `in_ch` channels."""
    info = medmnist.INFO[data]
    imgs = getattr(medmnist, info["python_class"])(split="test", download=True, size=28).imgs[:N_OOD]
    x = torch.from_numpy(imgs).float() / 255
    x = x[:, None] if x.ndim == 3 else x.permute(0, 3, 1, 2)
    if x.shape[1] != in_ch:  # grey -> RGB by repeating, RGB -> grey by averaging
        x = x.repeat(1, 3, 1, 1) if in_ch == 3 else x.mean(1, keepdim=True)
    return (x - .5) / .5


def ood_logits(data, method, seed):
    """{source: logits} (plus mc_ keys for dropout models), cached on disk."""
    loss, dropout = METHODS[method]
    name = run_name(data, loss, seed, dropout=dropout, select="nll")
    cache = f"runs/ood_{name}.npz"
    if os.path.exists(cache): return dict(np.load(cache))
    info = medmnist.INFO[data]
    model = resnet18(info["n_channels"], len(info["label"]), dropout).to(dev)
    model.load_state_dict(torch.load(f"runs/{name}.pt", map_location=dev))
    out = {}
    for src in DATASETS:
        if src == data: continue
        x = images(src, info["n_channels"])
        out[src] = predict_x(model, x, dev)
        if dropout: out[f"mc_{src}"] = predict_x(model, x, dev, MC)
    np.savez(cache + ".tmp.npz", **out); os.replace(cache + ".tmp.npz", cache)
    print("cached", cache, flush=True)
    return out


def auroc(in_logits, out_logits):
    s_in, s_out = probs(in_logits).max(1), probs(out_logits).max(1)
    return roc_auc_score(np.r_[np.ones(len(s_in)), np.zeros(len(s_out))], np.r_[s_in, s_out])


def main():
    res = {}  # (dataset, variant) -> list over seeds of {source: auroc}
    for data in DATASETS:
        for method, (loss, dropout) in METHODS.items():
            files = sorted(glob.glob(f"runs/{run_name(data, loss, '[0-9]', dropout=dropout, select='nll')}.npz"))
            seeds = [int(f[-5]) for f in files]
            runs = [np.load(f) for f in files]
            oods = [ood_logits(data, method, s) for s in seeds]
            srcs = [s for s in DATASETS if s != data]
            res[(data, NAMES[method])] = [{s: auroc(r["test_logits"], o[s]) for s in srcs} for r, o in zip(runs, oods)]
            if dropout:
                res[(data, "MC Dropout")] = [{s: auroc(r["mc_test_logits"], o[f"mc_{s}"]) for s in srcs}
                                             for r, o in zip(runs, oods)]
            else:  # ensemble: average member probabilities, then the log is a valid logit vector
                ens = lambda key, src=None: np.log(np.mean([probs(o[src] if src else r[key]) for r, o in zip(runs, oods)], 0) + 1e-12)
                res[(data, f"{NAMES[method]} ensemble")] = [{s: auroc(ens("test_logits"), ens(None, s)) for s in srcs}]

    L = ["# Out-of-distribution detection (max softmax probability; AUROC, 0.5 = chance, higher is better)", "",
         f"In-distribution = the model's own test set; OOD = the first {N_OOD} test images of each other dataset.", ""]
    for data in DATASETS:
        srcs = [s for s in DATASETS if s != data]
        L += [f"## Trained on {data}", "", "| Method | " + " | ".join(f"vs {s}" for s in srcs) + " | mean |",
              "|---|" + "---|" * (len(srcs) + 1)]
        for (d, v), rs in res.items():
            if d != data: continue
            cell = lambda vals: f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"
            L.append(f"| {v} | " + " | ".join(cell([r[s] for r in rs]) for s in srcs)
                     + f" | {cell([np.mean(list(r.values())) for r in rs])} |")
        L.append("")

    # pooled paired tests on mean OOD AUROC, by (dataset, seed)
    mean_auc = lambda d, v: [np.mean(list(r.values())) for r in res[(d, v)]]
    comps = [("Label smoothing", "Cross-entropy"), ("Focal loss", "Cross-entropy"),
             ("MC Dropout", "Cross-entropy"), ("MC Dropout", "Dropout 0.5")]
    rows, ps = [], []
    for A, B in comps:
        diff = np.concatenate([np.array(mean_auc(d, A)) - np.array(mean_auc(d, B)) for d in DATASETS])
        rows.append((A, B, len(diff), np.median(diff), int(np.sum(diff > 0)))); ps.append(wilcoxon(diff).pvalue)
    L += ["## Paired tests on mean OOD AUROC (pairs = dataset × seed; Wilcoxon, Holm)", "",
          "| A | vs B | n | Δ (A − B) | A wins | p (Holm) |", "|---|---|---|---|---|---|"]
    for (A, B, n, d, w), pa in zip(rows, holm(np.array(ps))):
        L.append(f"| {A} | {B} | {n} | {d:+.3f} | {w}/{n} | {pa:.4f}{' *' if pa < .05 else ''} |")

    out = "\n".join(L)
    open("results_ood.md", "w", encoding="utf-8").write(out + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
