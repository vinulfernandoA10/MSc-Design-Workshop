# Significance tests (pneumoniamnist, dermamnist, bloodmnist, organcmnist; checkpoint selected by val NLL)

Paired by (dataset, seed); two-sided Wilcoxon signed-rank; Holm-corrected within each metric.
*Wins* = pairs where A is better than B. Δ = median of (A − B).

## ECE, clean test (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 20 | +0.009 | 6/20 | 0.3118 | 0.7074 |
| Focal loss | Cross-entropy | 20 | -0.020 | 16/20 | 0.0973 | 0.4865 |
| MC Dropout | Cross-entropy | 20 | -0.004 | 14/20 | 0.2305 | 0.7074 |
| CE + temp. scaling | Cross-entropy | 20 | -0.018 | 18/20 | 0.0001 | 0.0005 * |
| LS + temp. scaling | Label smoothing | 20 | -0.020 | 13/20 | 0.1769 | 0.7074 |
| Focal + temp. scaling | Focal loss | 20 | -0.000 | 11/20 | 0.7285 | 0.7285 |
| MC Dropout + temp. scaling | MC Dropout | 20 | -0.015 | 18/20 | 0.0001 | 0.0004 * |

## ECE, severity 5 (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 20 | -0.105 | 18/20 | 0.0000 | 0.0001 * |
| Focal loss | Cross-entropy | 20 | -0.089 | 19/20 | 0.0000 | 0.0001 * |
| MC Dropout | Cross-entropy | 20 | -0.027 | 14/20 | 0.0153 | 0.0306 * |
| CE + temp. scaling | Cross-entropy | 20 | -0.060 | 16/20 | 0.0003 | 0.0008 * |
| LS + temp. scaling | Label smoothing | 20 | +0.067 | 0/20 | 0.0000 | 0.0000 * |
| Focal + temp. scaling | Focal loss | 20 | +0.004 | 8/20 | 0.0637 | 0.0637 |
| MC Dropout + temp. scaling | MC Dropout | 20 | -0.033 | 16/20 | 0.0002 | 0.0007 * |

## AUC, clean test (higher is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 20 | -0.002 | 2/20 | 0.0002 | 0.0012 * |
| Focal loss | Cross-entropy | 20 | -0.000 | 10/20 | 0.6742 | 0.6742 |
| MC Dropout | Cross-entropy | 20 | -0.000 | 9/20 | 0.3118 | 0.6236 |
| CE + temp. scaling | Cross-entropy | 20 | +0.000 | 9/20 | 0.0703 | 0.3516 |
| LS + temp. scaling | Label smoothing | 20 | -0.000 | 0/20 | 0.0004 | 0.0026 * |
| Focal + temp. scaling | Focal loss | 20 | +0.000 | 10/20 | 0.1556 | 0.4669 |
| MC Dropout + temp. scaling | MC Dropout | 20 | +0.000 | 12/20 | 0.0778 | 0.3516 |

