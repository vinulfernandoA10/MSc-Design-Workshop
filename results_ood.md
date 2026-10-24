# Out-of-distribution detection (max softmax probability; AUROC, 0.5 = chance, higher is better)

In-distribution = the model's own test set; OOD = the first 2000 test images of each other dataset.

## Trained on pneumoniamnist

| Method | vs dermamnist | vs bloodmnist | vs organcmnist | mean |
|---|---|---|---|---|
| Cross-entropy | 0.440 ± 0.066 | 0.834 ± 0.015 | 0.734 ± 0.093 | 0.669 ± 0.048 |
| Cross-entropy ensemble | 0.424 | 0.860 | 0.822 | 0.702 |
| Label smoothing | 0.571 ± 0.103 | 0.783 ± 0.051 | 0.744 ± 0.077 | 0.699 ± 0.053 |
| Label smoothing ensemble | 0.637 | 0.881 | 0.873 | 0.797 |
| Focal loss | 0.583 ± 0.081 | 0.842 ± 0.022 | 0.841 ± 0.019 | 0.755 ± 0.031 |
| Focal loss ensemble | 0.589 | 0.856 | 0.892 | 0.779 |
| Dropout 0.5 | 0.390 ± 0.058 | 0.795 ± 0.030 | 0.783 ± 0.036 | 0.656 ± 0.031 |
| MC Dropout | 0.387 ± 0.057 | 0.794 ± 0.032 | 0.787 ± 0.030 | 0.656 ± 0.029 |

## Trained on dermamnist

| Method | vs pneumoniamnist | vs bloodmnist | vs organcmnist | mean |
|---|---|---|---|---|
| Cross-entropy | 0.609 ± 0.136 | 0.435 ± 0.121 | 0.653 ± 0.088 | 0.566 ± 0.061 |
| Cross-entropy ensemble | 0.659 | 0.453 | 0.718 | 0.610 |
| Label smoothing | 0.628 ± 0.154 | 0.472 ± 0.215 | 0.649 ± 0.081 | 0.583 ± 0.053 |
| Label smoothing ensemble | 0.712 | 0.568 | 0.759 | 0.680 |
| Focal loss | 0.638 ± 0.142 | 0.366 ± 0.097 | 0.717 ± 0.067 | 0.574 ± 0.080 |
| Focal loss ensemble | 0.602 | 0.357 | 0.729 | 0.563 |
| Dropout 0.5 | 0.562 ± 0.169 | 0.525 ± 0.099 | 0.541 ± 0.091 | 0.543 ± 0.047 |
| MC Dropout | 0.562 ± 0.169 | 0.546 ± 0.093 | 0.550 ± 0.091 | 0.553 ± 0.048 |

## Trained on bloodmnist

| Method | vs pneumoniamnist | vs dermamnist | vs organcmnist | mean |
|---|---|---|---|---|
| Cross-entropy | 0.181 ± 0.022 | 0.529 ± 0.051 | 0.189 ± 0.018 | 0.300 ± 0.014 |
| Cross-entropy ensemble | 0.161 | 0.754 | 0.164 | 0.360 |
| Label smoothing | 0.130 ± 0.074 | 0.654 ± 0.062 | 0.062 ± 0.026 | 0.282 ± 0.027 |
| Label smoothing ensemble | 0.133 | 0.756 | 0.073 | 0.321 |
| Focal loss | 0.097 ± 0.059 | 0.409 ± 0.105 | 0.061 ± 0.027 | 0.189 ± 0.047 |
| Focal loss ensemble | 0.124 | 0.694 | 0.072 | 0.297 |
| Dropout 0.5 | 0.246 ± 0.092 | 0.480 ± 0.106 | 0.193 ± 0.020 | 0.306 ± 0.063 |
| MC Dropout | 0.224 ± 0.112 | 0.541 ± 0.138 | 0.166 ± 0.029 | 0.310 ± 0.076 |

## Trained on organcmnist

| Method | vs pneumoniamnist | vs dermamnist | vs bloodmnist | mean |
|---|---|---|---|---|
| Cross-entropy | 0.897 ± 0.011 | 0.889 ± 0.026 | 0.918 ± 0.019 | 0.901 ± 0.013 |
| Cross-entropy ensemble | 0.915 | 0.891 | 0.957 | 0.921 |
| Label smoothing | 0.886 ± 0.026 | 0.901 ± 0.009 | 0.936 ± 0.012 | 0.908 ± 0.010 |
| Label smoothing ensemble | 0.919 | 0.914 | 0.948 | 0.927 |
| Focal loss | 0.889 ± 0.042 | 0.908 ± 0.017 | 0.946 ± 0.007 | 0.914 ± 0.020 |
| Focal loss ensemble | 0.923 | 0.915 | 0.970 | 0.936 |
| Dropout 0.5 | 0.888 ± 0.015 | 0.894 ± 0.030 | 0.943 ± 0.006 | 0.908 ± 0.011 |
| MC Dropout | 0.884 ± 0.016 | 0.891 ± 0.029 | 0.944 ± 0.005 | 0.906 ± 0.012 |

## Paired tests on mean OOD AUROC (pairs = dataset × seed; Wilcoxon, Holm)

| A | vs B | n | Δ (A − B) | A wins | p (Holm) |
|---|---|---|---|---|---|
| Label smoothing | Cross-entropy | 20 | -0.001 | 10/20 | 1.0000 |
| Focal loss | Cross-entropy | 20 | +0.007 | 11/20 | 1.0000 |
| MC Dropout | Cross-entropy | 20 | -0.000 | 10/20 | 1.0000 |
| MC Dropout | Dropout 0.5 | 20 | +0.000 | 10/20 | 1.0000 |
