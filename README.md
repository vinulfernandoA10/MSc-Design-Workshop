# How Reliable Is Model Confidence Under Distribution Shift?

M.Sc. AI thesis: calibration and uncertainty estimation for medical image classification on [MedMNIST v2](https://medmnist.com/).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Tested on a GTX 1050 Ti (4 GB): ResNet-18 at 28px uses ~700 MiB, ~10 s/epoch on PneumoniaMNIST.

## Usage

```bash
# train baseline (MedMNIST v2 recipe: Adam 1e-3, batch 128, 100 epochs); saves model + val/test logits to runs/
python train_baseline.py --data pneumoniamnist --epochs 100 --seed 0

# temperature scaling on the saved logits (no retraining)
python temp_scale.py runs/pneumoniamnist_28_ce_s0.npz
```

## Results so far (PneumoniaMNIST, 28px, seed 0)

| | Acc | AUC | ECE | NLL |
|---|---|---|---|---|
| MedMNIST v2 paper (ResNet-18) | 0.854 | 0.944 | – | – |
| Baseline, 100 epochs | 0.843 | 0.937 | 0.146 | 1.387 |
| + Temperature scaling (T = 3.15) | 0.843 | 0.939 | 0.113 | 0.488 |
