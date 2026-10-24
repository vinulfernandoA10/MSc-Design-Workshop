"""Week-1 baseline: ResNet-18 on a MedMNIST dataset, reports accuracy, AUC, ECE, NLL.

Saves val/test logits so post-hoc methods (temperature scaling, ensembles) need no retraining.
Recipe follows the MedMNIST v2 paper: Adam lr 1e-3, batch 128, 100 epochs, lr x0.1 at 50 and 75.

    python train_baseline.py --data pneumoniamnist --epochs 100 --seed 0 [--loss ce|ls|focal]
"""
import argparse, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import medmnist
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from corrupt import SEVERITY, corrupt


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error: weighted gap between confidence and accuracy per confidence bin."""
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return sum(abs((pred[m] == labels[m]).mean() - conf[m].mean()) * m.mean()
               for b in range(n_bins) if (m := bins == b).any())


def selective(probs, labels, coverage=0.8):
    """Selective prediction (RQ3): accept the most confident cases first, refer the rest.

    aurc  = area under the risk-coverage curve (mean error rate over all coverage levels);
    eaurc = aurc minus the best possible aurc for this many errors (Geifman et al., 2019), i.e. how
            well confidence ranks correct above wrong predictions, independent of overall accuracy;
    acc80 = accuracy on the `coverage` most confident fraction of cases.
    """
    conf, err = probs.max(1), (probs.argmax(1) != labels).astype(float)
    n, k = len(err), np.arange(1, len(err) + 1)
    risk = np.cumsum(err[np.argsort(-conf, kind="stable")]) / k
    best = np.maximum(0, k - (n - err.sum())) / k  # oracle: every error ranked last
    return dict(aurc=risk.mean(), eaurc=risk.mean() - best.mean(), acc80=1 - risk[int(coverage * n) - 1])


def focal_loss(logits, y, gamma=3.0):
    """Focal loss (Lin et al., 2017); gamma=3 is the calibration setting of Mukhoti et al. (2020)."""
    logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** gamma * logp).mean()


LOSSES = {
    "ce": F.cross_entropy,
    "ls": lambda z, y: F.cross_entropy(z, y, label_smoothing=0.1),  # Szegedy et al. (2016); 0.1 is the standard value
    "focal": focal_loss,
}


def metrics(logits, labels):
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    auc = roc_auc_score(labels, probs[:, 1]) if probs.shape[1] == 2 else roc_auc_score(labels, probs, multi_class="ovr")
    nll = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    # bacc = mean per-class recall: unlike accuracy, it does not reward collapsing onto the majority class
    return dict(acc=(probs.argmax(1) == labels).mean(), bacc=balanced_accuracy_score(labels, probs.argmax(1)),
                auc=auc, ece=ece(probs, labels), nll=nll, **selective(probs, labels))


def resnet18(in_ch, n_cls, p_drop=0.0):
    m = torchvision.models.resnet18(num_classes=n_cls)
    m.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)  # small-image stem: 28px input would vanish under the 7x7/stride-2 + maxpool stem
    m.maxpool = nn.Identity()
    if p_drop:  # dropout before the classifier, as in Gal & Ghahramani's (2016) CNN experiments
        m.fc = nn.Sequential(nn.Dropout(p_drop), m.fc)
    return m


METHODS = {  # method -> (loss, dropout); shared by run_all.py and analyze.py
    "ce": ("ce", 0.0), "ls": ("ls", 0.0), "focal": ("focal", 0.0), "mcdo": ("ce", 0.5)}


def run_name(data, loss, seed, size=28, dropout=0.0, select="auc"):
    """File stem shared by train_baseline.py, run_all.py and analyze.py."""
    return f"{data}_{size}_{loss}{f'_do{dropout:g}' if dropout else ''}{'_nll' if select == 'nll' else ''}_s{seed}"


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = [(model(x.to(dev)).cpu(), y.view(-1)) for x, y in loader]
    return torch.cat([o for o, _ in out]).numpy(), torch.cat([y for _, y in out]).numpy()


@torch.no_grad()
def predict_x(model, x, dev, mc=0):
    """Logits for a tensor of normalised images; with mc > 0, log of the MC-Dropout-averaged probabilities."""
    model.eval()
    if not mc:
        return torch.cat([model(b.to(dev)).cpu() for b in x.split(512)]).numpy()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()  # keep dropout active at test time
    p = sum(torch.cat([model(b.to(dev)).softmax(1).cpu() for b in x.split(512)]) for _ in range(mc)) / mc
    model.eval()
    return p.clamp_min(1e-12).log().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pneumoniamnist")
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--loss", default="ce", choices=LOSSES)
    p.add_argument("--select", default="auc", choices=["auc", "nll"], help="validation metric for checkpoint selection")
    p.add_argument("--dropout", type=float, default=0.0, help="> 0 adds dropout and saves MC Dropout predictions")
    p.add_argument("--mc", type=int, default=20, help="MC Dropout forward passes")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    info = medmnist.INFO[a.data]
    assert info["task"] != "multi-label, binary-class", "multi-label sets (ChestMNIST) need a sigmoid head; out of scope"
    DS = getattr(medmnist, info["python_class"])
    tf = T.Compose([T.ToTensor(), T.Normalize([.5], [.5])])
    load = lambda split, shuf: torch.utils.data.DataLoader(
        DS(split=split, transform=tf, download=True, size=a.size), batch_size=a.bs, shuffle=shuf, num_workers=0)
    tr, va, te = load("train", True), load("val", False), load("test", False)

    model = resnet18(info["n_channels"], len(info["label"]), a.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.epochs * .5), int(a.epochs * .75)], 0.1)

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.out}/{run_name(a.data, a.loss, a.seed, a.size, a.dropout, a.select)}"
    ckpt = f"{tag}_resume.pt"  # per-epoch state so an interrupted run continues instead of restarting
    best_score, best_ep, start, best = -np.inf, 0, 0, None
    if os.path.exists(ckpt):
        s = torch.load(ckpt, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        best_score, best_ep, start, best = s["best_score"], s["best_ep"], s["epoch"], s["best"]
        torch.set_rng_state(s["rng"])
        if dev == "cuda": torch.cuda.set_rng_state(s["cuda_rng"])
        print(f"resuming after epoch {start}")
    for ep in range(start, a.epochs):
        model.train(); t0 = time.time()
        for x, y in tr:
            loss = LOSSES[a.loss](model(x.to(dev)), y.view(-1).long().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        vl, vy = predict(model, va, dev)
        vm = metrics(vl, vy.astype(np.int64))
        score = vm["auc"] if a.select == "auc" else -vm["nll"]  # AUC: MedMNIST v2 protocol; NLL: tracks overconfidence
        if score > best_score:
            best_score, best_ep = score, ep + 1
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        best=best, best_score=best_score, best_ep=best_ep, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if dev == "cuda" else None), ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)  # atomic: a kill mid-save leaves the previous checkpoint intact
        print(f"epoch {ep + 1}/{a.epochs} loss {loss.item():.4f} val_auc {vm['auc']:.4f} val_nll {vm['nll']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best)
    print(f"best epoch {best_ep} (by val {a.select})")
    grab = lambda loader: (torch.cat([x for x, _ in loader]), torch.cat([y.view(-1) for _, y in loader]).numpy())
    (xv, vy), (xt, ty) = grab(va), grab(te)
    out = dict(val_labels=vy, test_labels=ty, best_epoch=best_ep)

    def evaluate(key, x):
        out[f"{key}_logits"] = predict_x(model, x, dev)
        if a.dropout: out[f"mc_{key}_logits"] = predict_x(model, x, dev, a.mc)

    evaluate("val", xv); evaluate("test", xt)
    for c in SEVERITY:  # corrupted copies of the test set, for the distribution-shift experiments
        for s in range(1, 6):
            evaluate(f"c_{c}_{s}", (corrupt(xt * .5 + .5, c, s) - .5) / .5)
    np.savez(f"{tag}.npz", **out)
    torch.save(model.state_dict(), f"{tag}.pt")
    os.remove(ckpt)
    ty = ty.astype(np.int64)
    print("test:", {k: round(float(v), 4) for k, v in metrics(out["test_logits"], ty).items()})
    if a.dropout: print("test MC Dropout:", {k: round(float(v), 4) for k, v in metrics(out["mc_test_logits"], ty).items()})
    if dev == "cuda": print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    # self-check: a perfectly calibrated toy case gives ECE 0, an overconfident one gives the gap
    y = np.array([0, 0, 1, 1]); assert ece(np.array([[1., 0], [1, 0], [0, 1], [0, 1]]), y) == 0
    assert abs(ece(np.array([[.9, .1]] * 4), y) - 0.4) < 1e-9
    z, t = torch.randn(8, 3), torch.randint(0, 3, (8,))  # focal loss with gamma=0 reduces to cross-entropy
    assert torch.allclose(focal_loss(z, t, gamma=0), F.cross_entropy(z, t))
    # selective prediction: errors ranked last gives e-aurc 0; errors ranked first is worse
    lab = np.array([0, 0, 0, 1]); good = np.array([[.9, .1], [.8, .2], [.7, .3], [.6, .4]])  # last is wrong, least confident
    bad = np.array([[.6, .4], [.7, .3], [.8, .2], [.95, .05]])  # the wrong prediction is the most confident
    s_good, s_bad = selective(good, lab), selective(bad, lab)
    assert abs(s_good["eaurc"]) < 1e-12 and s_bad["eaurc"] > 0.3 and s_good["acc80"] == 1.0
    main()
