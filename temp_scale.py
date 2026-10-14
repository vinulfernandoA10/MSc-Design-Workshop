"""Temperature scaling (Guo et al., 2017): fit one scalar T on val logits, apply to test logits.

    python temp_scale.py runs/pneumoniamnist_28_ce_s0.npz
"""
import sys
import numpy as np
import torch, torch.nn.functional as F
from train_baseline import metrics


def fit_temperature(logits, labels):
    logits, labels = torch.from_numpy(logits).float(), torch.from_numpy(labels).long()
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    def step():
        opt.zero_grad(); loss = F.cross_entropy(logits / log_t.exp(), labels); loss.backward(); return loss
    opt.step(step)
    return log_t.exp().item()


if __name__ == "__main__":
    d = np.load(sys.argv[1])
    vy, ty = d["val_labels"].astype(np.int64), d["test_labels"].astype(np.int64)
    T = fit_temperature(d["val_logits"], vy)
    r = lambda m: {k: round(float(v), 4) for k, v in m.items()}
    print(f"T = {T:.3f}")
    print("before:", r(metrics(d["test_logits"], ty)))
    print("after: ", r(metrics(d["test_logits"] / T, ty)))
