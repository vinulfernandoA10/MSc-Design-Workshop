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
# all methods (cross-entropy, label smoothing, focal loss, MC Dropout) x seeds 0-4; resumable
python run_all.py --data pneumoniamnist --epochs 50

# summary tables: clean test set, plus ECE / accuracy under 5 corruptions x 5 severities
python analyze.py --data pneumoniamnist
```

| File | Purpose |
|---|---|
| `train_baseline.py` | One training run: ResNet-18, `--loss ce/ls/focal`, `--dropout`, `--select auc/nll`; saves logits for val, test, MC Dropout and 25 corrupted test sets |
| `corrupt.py` | Test-time corruptions (noise, blur, contrast, brightness, JPEG) at 5 severities, after CIFAR-10-C |
| `temp_scale.py` | Temperature scaling on saved logits |
| `run_all.py` | Runs every method x seed; skips finished runs |
| `analyze.py` | Mean ± std over seeds, temperature scaling, ensembles, shift tables |

## Results

Main experiments (4 methods x 5 seeds, checkpoint selected by val NLL, 50 epochs, clean + 25 corrupted test sets):
[PneumoniaMNIST](results_pneumoniamnist_nll.md) · [DermaMNIST](results_dermamnist_nll.md) · [BloodMNIST](results_bloodmnist_nll.md) ·
[pooled significance tests](significance_pneumoniamnist_dermamnist_bloodmnist_nll.md) (`python significance.py --data pneumoniamnist dermamnist bloodmnist`).

Earlier: [results_pneumoniamnist_auc.md](results_pneumoniamnist_auc.md): 5 seeds, checkpoint selected by val AUC (MedMNIST protocol), clean test only.

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
