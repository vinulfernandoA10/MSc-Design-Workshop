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

## E-AURC, clean test (RQ3) (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 20 | +0.007 | 1/20 | 0.0000 | 0.0001 * |
| Focal loss | Cross-entropy | 20 | +0.001 | 8/20 | 0.4304 | 0.4304 |
| MC Dropout | Cross-entropy | 20 | +0.000 | 6/20 | 0.0637 | 0.3186 |
| CE + temp. scaling | Cross-entropy | 20 | +0.000 | 7/20 | 0.0787 | 0.3186 |
| LS + temp. scaling | Label smoothing | 20 | -0.000 | 16/20 | 0.0004 | 0.0021 * |
| Focal + temp. scaling | Focal loss | 20 | -0.000 | 11/20 | 0.2115 | 0.4230 |
| MC Dropout + temp. scaling | MC Dropout | 20 | +0.000 | 7/20 | 0.1024 | 0.3186 |

## E-AURC, severity 5 (RQ3) (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 20 | +0.017 | 3/20 | 0.0023 | 0.0163 * |
| Focal loss | Cross-entropy | 20 | +0.007 | 8/20 | 0.8695 | 1.0000 |
| MC Dropout | Cross-entropy | 20 | -0.002 | 11/20 | 0.9854 | 1.0000 |
| CE + temp. scaling | Cross-entropy | 20 | -0.000 | 12/20 | 0.0583 | 0.3495 |
| LS + temp. scaling | Label smoothing | 20 | -0.000 | 13/20 | 0.2024 | 0.8098 |
| Focal + temp. scaling | Focal loss | 20 | +0.000 | 7/20 | 0.5349 | 1.0000 |
| MC Dropout + temp. scaling | MC Dropout | 20 | -0.000 | 14/20 | 0.0583 | 0.3495 |

