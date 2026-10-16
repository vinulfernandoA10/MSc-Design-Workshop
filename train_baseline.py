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
from sklearn.metrics import roc_auc_score


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error: weighted gap between confidence and accuracy per confidence bin."""
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return sum(abs((pred[m] == labels[m]).mean() - conf[m].mean()) * m.mean()
               for b in range(n_bins) if (m := bins == b).any())


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
    return dict(acc=(probs.argmax(1) == labels).mean(), auc=auc, ece=ece(probs, labels), nll=nll)


def resnet18(in_ch, n_cls):
    m = torchvision.models.resnet18(num_classes=n_cls)
    m.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)  # small-image stem: 28px input would vanish under the 7x7/stride-2 + maxpool stem
    m.maxpool = nn.Identity()
    return m


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = [(model(x.to(dev)).cpu(), y.view(-1)) for x, y in loader]
    return torch.cat([o for o, _ in out]).numpy(), torch.cat([y for _, y in out]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pneumoniamnist")
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--loss", default="ce", choices=LOSSES)
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

    model = resnet18(info["n_channels"], len(info["label"])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.epochs * .5), int(a.epochs * .75)], 0.1)

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.out}/{a.data}_{a.size}_{a.loss}_s{a.seed}"
    ckpt = f"{tag}_resume.pt"  # per-epoch state so an interrupted run continues instead of restarting
    best_auc, best_ep, start, best = -1, 0, 0, None
    if os.path.exists(ckpt):
        s = torch.load(ckpt, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        best_auc, best_ep, start, best = s["best_auc"], s["best_ep"], s["epoch"], s["best"]
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
        val_auc = metrics(vl, vy.astype(np.int64))["auc"]
        if val_auc > best_auc:  # model selection by val AUC, as in the MedMNIST v2 paper
            best_auc, best_ep = val_auc, ep + 1
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        best=best, best_auc=best_auc, best_ep=best_ep, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if dev == "cuda" else None), ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)  # atomic: a kill mid-save leaves the previous checkpoint intact
        print(f"epoch {ep + 1}/{a.epochs} loss {loss.item():.4f} val_auc {val_auc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best)
    print(f"best epoch {best_ep} (val AUC {best_auc:.4f})")
    vl, vy = predict(model, va, dev); tl, ty = predict(model, te, dev)
    np.savez(f"{tag}.npz", val_logits=vl, val_labels=vy, test_logits=tl, test_labels=ty, best_epoch=best_ep)
    torch.save(model.state_dict(), f"{tag}.pt")
    os.remove(ckpt)
    print("test:", {k: round(float(v), 4) for k, v in metrics(tl, ty.astype(np.int64)).items()})
    if dev == "cuda": print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    # self-check: a perfectly calibrated toy case gives ECE 0, an overconfident one gives the gap
    y = np.array([0, 0, 1, 1]); assert ece(np.array([[1., 0], [1, 0], [0, 1], [0, 1]]), y) == 0
    assert abs(ece(np.array([[.9, .1]] * 4), y) - 0.4) < 1e-9
    z, t = torch.randn(8, 3), torch.randint(0, 3, (8,))  # focal loss with gamma=0 reduces to cross-entropy
    assert torch.allclose(focal_loss(z, t, gamma=0), F.cross_entropy(z, t))
    main()
