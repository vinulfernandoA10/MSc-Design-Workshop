# Significance tests (pneumoniamnist, dermamnist, bloodmnist; checkpoint selected by val NLL)

Paired by (dataset, seed); two-sided Wilcoxon signed-rank; Holm-corrected within each metric.
*Wins* = pairs where A is better than B. Δ = median of (A − B).

## ECE, clean test (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 15 | +0.020 | 6/15 | 0.4887 | 1.0000 |
| Focal loss | Cross-entropy | 15 | -0.018 | 11/15 | 0.4212 | 1.0000 |
| MC Dropout | Cross-entropy | 15 | -0.004 | 9/15 | 0.7197 | 1.0000 |
| CE + temp. scaling | Cross-entropy | 15 | -0.013 | 13/15 | 0.0026 | 0.0157 * |
| LS + temp. scaling | Label smoothing | 15 | -0.010 | 8/15 | 0.3591 | 1.0000 |
| Focal + temp. scaling | Focal loss | 15 | -0.003 | 9/15 | 0.3894 | 1.0000 |
| MC Dropout + temp. scaling | MC Dropout | 15 | -0.014 | 13/15 | 0.0020 | 0.0141 * |

## ECE, severity 5 (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 15 | -0.056 | 13/15 | 0.0003 | 0.0018 * |
| Focal loss | Cross-entropy | 15 | -0.060 | 14/15 | 0.0004 | 0.0021 * |
| MC Dropout | Cross-entropy | 15 | -0.034 | 10/15 | 0.0302 | 0.0603 |
| CE + temp. scaling | Cross-entropy | 15 | -0.016 | 11/15 | 0.0084 | 0.0251 * |
| LS + temp. scaling | Label smoothing | 15 | +0.043 | 0/15 | 0.0001 | 0.0004 * |
| Focal + temp. scaling | Focal loss | 15 | +0.006 | 6/15 | 0.1205 | 0.1205 |
| MC Dropout + temp. scaling | MC Dropout | 15 | -0.023 | 11/15 | 0.0054 | 0.0215 * |

## AUC, clean test (higher is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 15 | -0.011 | 1/15 | 0.0009 | 0.0060 * |
| Focal loss | Cross-entropy | 15 | -0.000 | 7/15 | 0.6788 | 1.0000 |
| MC Dropout | Cross-entropy | 15 | +0.000 | 8/15 | 0.7197 | 1.0000 |
| CE + temp. scaling | Cross-entropy | 15 | +0.000 | 4/15 | 1.0000 | 1.0000 |
| LS + temp. scaling | Label smoothing | 15 | -0.000 | 0/15 | 0.0033 | 0.0201 * |
| Focal + temp. scaling | Focal loss | 15 | +0.000 | 8/15 | 0.2026 | 1.0000 |
| MC Dropout + temp. scaling | MC Dropout | 15 | +0.000 | 7/15 | 0.5067 | 1.0000 |

