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

## Results

5-seed summary (mean ± std, ensembles): [results_pneumoniamnist.md](results_pneumoniamnist.md),
produced by `python run_all.py` then `python analyze.py`.

### Seed 0 only (PneumoniaMNIST, 28px)

| | Acc | AUC | ECE | NLL |
|---|---|---|---|---|
| MedMNIST v2 paper (ResNet-18) | 0.854 | 0.944 | – | – |
| Baseline, best val-AUC checkpoint (epoch 5) | 0.894 | 0.950 | 0.075 | 0.437 |
| + Temperature scaling (T = 1.93) | 0.894 | 0.950 | 0.039 | 0.286 |
| Label smoothing 0.1 (epoch 3) | 0.873 | 0.889 | 0.040 | 0.357 |
| + Temperature scaling (T = 0.59) | 0.873 | 0.889 | 0.085 | 0.462 |
| Focal loss γ=3 (epoch 4) | 0.861 | 0.954 | 0.051 | 0.351 |
| + Temperature scaling (T = 0.47) | 0.861 | 0.954 | 0.098 | 0.521 |
| Baseline, last epoch (100) | 0.843 | 0.937 | 0.146 | 1.387 |
| + Temperature scaling (T = 3.15) | 0.843 | 0.939 | 0.113 | 0.488 |

Note: PneumoniaMNIST has a built-in label shift — pneumonia is 74.2% of val but 62.5% of test —
so temperatures fitted on val do not transfer cleanly to test.
