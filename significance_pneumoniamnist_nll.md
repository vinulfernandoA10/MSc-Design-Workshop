# Significance tests (pneumoniamnist; checkpoint selected by val NLL)

Paired by (dataset, seed); two-sided Wilcoxon signed-rank; Holm-corrected within each metric.
*Wins* = pairs where A is better than B. Δ = median of (A − B).

## ECE, clean test (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 5 | -0.043 | 5/5 | 0.0625 | 0.4375 |
| Focal loss | Cross-entropy | 5 | -0.028 | 5/5 | 0.0625 | 0.4375 |
| MC Dropout | Cross-entropy | 5 | +0.005 | 2/5 | 1.0000 | 1.0000 |
| CE + temp. scaling | Cross-entropy | 5 | -0.010 | 3/5 | 0.3125 | 0.9375 |
| LS + temp. scaling | Label smoothing | 5 | +0.041 | 0/5 | 0.0625 | 0.4375 |
| Focal + temp. scaling | Focal loss | 5 | +0.005 | 1/5 | 0.1875 | 0.7500 |
| MC Dropout + temp. scaling | MC Dropout | 5 | -0.016 | 3/5 | 0.3125 | 0.9375 |

## ECE, severity 5 (lower is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 5 | -0.056 | 5/5 | 0.0625 | 0.4375 |
| Focal loss | Cross-entropy | 5 | -0.060 | 5/5 | 0.0625 | 0.4375 |
| MC Dropout | Cross-entropy | 5 | -0.034 | 4/5 | 0.1250 | 0.5000 |
| CE + temp. scaling | Cross-entropy | 5 | -0.011 | 3/5 | 0.3125 | 0.6250 |
| LS + temp. scaling | Label smoothing | 5 | +0.043 | 0/5 | 0.0625 | 0.4375 |
| Focal + temp. scaling | Focal loss | 5 | +0.006 | 1/5 | 0.1875 | 0.5625 |
| MC Dropout + temp. scaling | MC Dropout | 5 | -0.016 | 3/5 | 0.3125 | 0.6250 |

## AUC, clean test (higher is better)

| A | vs B | n | Δ (A − B) | A wins | p | p (Holm) |
|---|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 5 | -0.091 | 0/5 | 0.0625 | 0.4375 |
| Focal loss | Cross-entropy | 5 | +0.001 | 4/5 | 0.3125 | 1.0000 |
| MC Dropout | Cross-entropy | 5 | -0.001 | 2/5 | 0.8125 | 1.0000 |
| CE + temp. scaling | Cross-entropy | 5 | +0.000 | 0/5 | 1.0000 | 1.0000 |
| LS + temp. scaling | Label smoothing | 5 | +0.000 | 0/5 | 1.0000 | 1.0000 |
| Focal + temp. scaling | Focal loss | 5 | +0.000 | 0/5 | 1.0000 | 1.0000 |
| MC Dropout + temp. scaling | MC Dropout | 5 | +0.000 | 2/5 | 0.5000 | 1.0000 |

> With n = 5 pairs, a two-sided Wilcoxon test cannot reach p < 0.05 (smallest possible p = 0.0625 at n = 5). Pool more datasets for significance.
