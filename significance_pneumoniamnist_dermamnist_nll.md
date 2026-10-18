# Significance tests (pneumoniamnist, dermamnist; checkpoint selected by val NLL)

Paired by (dataset, seed); two-sided Wilcoxon signed-rank; Holm-corrected within each metric.
*Wins* = pairs where A is better than B. Δ = median of (A − B).

## ECE, clean test (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 10 | -0.019 | 6/10 | 0.4316 | 1.0000 |
| Focal loss | Cross-entropy | 10 | -0.022 | 6/10 | 1.0000 | 1.0000 |
| MC Dropout | Cross-entropy | 10 | +0.003 | 4/10 | 0.9219 | 1.0000 |
| CE + temp. scaling | Cross-entropy | 10 | -0.005 | 8/10 | 0.0840 | 0.5039 |
| LS + temp. scaling | Label smoothing | 10 | +0.021 | 3/10 | 0.1309 | 0.6543 |
| Focal + temp. scaling | Focal loss | 10 | +0.003 | 4/10 | 0.9219 | 1.0000 |
| MC Dropout + temp. scaling | MC Dropout | 10 | -0.009 | 8/10 | 0.0488 | 0.3418 |

## ECE, severity 5 (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 10 | -0.035 | 8/10 | 0.0098 | 0.0586 |
| Focal loss | Cross-entropy | 10 | -0.041 | 9/10 | 0.0098 | 0.0586 |
| MC Dropout | Cross-entropy | 10 | -0.018 | 6/10 | 0.2754 | 0.6973 |
| CE + temp. scaling | Cross-entropy | 10 | -0.004 | 6/10 | 0.2324 | 0.6973 |
| LS + temp. scaling | Label smoothing | 10 | +0.034 | 0/10 | 0.0020 | 0.0137 * |
| Focal + temp. scaling | Focal loss | 10 | -0.004 | 6/10 | 0.7695 | 0.7695 |
| MC Dropout + temp. scaling | MC Dropout | 10 | -0.010 | 6/10 | 0.1602 | 0.6406 |

## AUC, clean test (higher is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 10 | -0.055 | 1/10 | 0.0039 | 0.0273 * |
| Focal loss | Cross-entropy | 10 | -0.000 | 5/10 | 0.6953 | 1.0000 |
| MC Dropout | Cross-entropy | 10 | -0.000 | 5/10 | 0.8457 | 1.0000 |
| CE + temp. scaling | Cross-entropy | 10 | +0.000 | 3/10 | 0.4375 | 1.0000 |
| LS + temp. scaling | Label smoothing | 10 | -0.000 | 0/10 | 0.0312 | 0.1875 |
| Focal + temp. scaling | Focal loss | 10 | +0.000 | 3/10 | 0.6250 | 1.0000 |
| MC Dropout + temp. scaling | MC Dropout | 10 | +0.000 | 5/10 | 0.3125 | 1.0000 |

