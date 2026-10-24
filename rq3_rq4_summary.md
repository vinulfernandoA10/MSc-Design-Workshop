# RQ3 and RQ4 summary (pneumoniamnist, dermamnist, bloodmnist, organcmnist; checkpoint selected by val NLL)

## RQ3: is a better-calibrated model better at selective prediction?

Spearman ρ between ECE and E-AURC over all single-model runs (4 methods × ± temperature scaling × 5 seeds).
ρ near 0 = calibration and the ability to rank right above wrong are unrelated.

| Dataset | runs | ρ clean (p) | ρ severity 5 (p) |
|---|---|---|---|
| pneumoniamnist | 40 | +0.14 (0.404) | +0.40 (0.011) |
| dermamnist | 40 | +0.27 (0.095) | +0.33 (0.038) |
| bloodmnist | 40 | +0.08 (0.635) | -0.23 (0.157) |
| organcmnist | 40 | +0.37 (0.020) | -0.11 (0.511) |

## RQ4: cost vs benefit (mean over datasets of the per-dataset seed mean)

| Method | training cost | test passes | ECE clean | ECE sev 5 | E-AURC clean | E-AURC sev 5 | Bal. acc sev 5 |
|---|---|---|---|---|---|---|---|
| Cross-entropy | 1× | 1× | 0.062 | 0.289 | 0.026 | 0.135 | 0.489 |
| CE + temp. scaling | 1× | 1× | 0.045 | 0.245 | 0.026 | 0.135 | 0.489 |
| Label smoothing | 1× | 1× | 0.070 | 0.194 | 0.060 | 0.174 | 0.487 |
| Focal loss | 1× | 1× | 0.050 | 0.212 | 0.026 | 0.136 | 0.491 |
| MC Dropout | 1× | 20× | 0.058 | 0.264 | 0.027 | 0.136 | 0.491 |
| CE ensemble (5) | 5× | 5× | 0.041 | 0.216 | 0.022 | 0.132 | 0.502 |
| LS ensemble (5) | 5× | 5× | 0.076 | 0.161 | 0.051 | 0.148 | 0.500 |
| Focal ensemble (5) | 5× | 5× | 0.056 | 0.179 | 0.022 | 0.124 | 0.506 |

## RQ4: is a single model as good as the cross-entropy ensemble?

Paired by (dataset, seed) against that dataset's CE ensemble; two-sided Wilcoxon, Holm within each metric.

### ECE sev 5 (lower is better)

| Single model | vs | n | Δ (single − ensemble) | single wins | p (Holm) |
|---|---|---|---|---|---|
| Cross-entropy | CE ensemble | 20 | +0.065 | 3/20 | 0.0002 * |
| CE + temp. scaling | CE ensemble | 20 | +0.020 | 7/20 | 0.0362 * |
| Label smoothing | CE ensemble | 20 | -0.027 | 13/20 | 0.0592 |
| Focal loss | CE ensemble | 20 | -0.007 | 11/20 | 0.5217 |
| MC Dropout | CE ensemble | 20 | +0.046 | 4/20 | 0.0079 * |

### E-AURC sev 5 (lower is better)

| Single model | vs | n | Δ (single − ensemble) | single wins | p (Holm) |
|---|---|---|---|---|---|
| Cross-entropy | CE ensemble | 20 | +0.007 | 8/20 | 0.9221 |
| CE + temp. scaling | CE ensemble | 20 | +0.006 | 8/20 | 0.9221 |
| Label smoothing | CE ensemble | 20 | +0.017 | 2/20 | 0.0008 * |
| Focal loss | CE ensemble | 20 | +0.002 | 8/20 | 0.9221 |
| MC Dropout | CE ensemble | 20 | +0.005 | 7/20 | 0.9221 |

