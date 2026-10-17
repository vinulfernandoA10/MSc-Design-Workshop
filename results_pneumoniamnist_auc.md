# Results: pneumoniamnist (28px, checkpoint selected by val AUC)

## Clean test set (mean ± std over seeds)

| Method | Runs | Acc | AUC | ECE ↓ | NLL ↓ |
|---|---|---|---|---|---|
| Cross-entropy | 5 | 0.865 ± 0.018 | 0.939 ± 0.013 | 0.119 ± 0.024 | 0.931 ± 0.280 |
| + temp. scaling (T = 2.19) | 5 | 0.865 ± 0.018 | 0.939 ± 0.013 | 0.091 ± 0.029 | 0.478 ± 0.139 |
| 5-seed ensemble | 5 | 0.872 | 0.953 | 0.082 | 0.524 |
| *best epochs: [5, 13, 20, 30, 9]* | | | | | |
| Label smoothing | 5 | 0.846 ± 0.025 | 0.893 ± 0.028 | 0.076 ± 0.021 | 0.424 ± 0.042 |
| + temp. scaling (T = 0.64) | 5 | 0.846 ± 0.025 | 0.893 ± 0.028 | 0.121 ± 0.022 | 0.557 ± 0.069 |
| 5-seed ensemble | 5 | 0.851 | 0.934 | 0.051 | 0.383 |
| *best epochs: [3, 50, 2, 3, 15]* | | | | | |
| Focal loss | 5 | 0.848 ± 0.025 | 0.947 ± 0.012 | 0.088 ± 0.025 | 0.423 ± 0.078 |
| + temp. scaling (T = 0.64) | 5 | 0.848 ± 0.025 | 0.947 ± 0.012 | 0.103 ± 0.027 | 0.519 ± 0.091 |
| 5-seed ensemble | 5 | 0.841 | 0.963 | 0.054 | 0.351 |
| *best epochs: [4, 2, 17, 17, 50]* | | | | | |
