# Detection Benchmark Results

Primary metric: **Test AP_50** (PASCAL VOC style, appropriate for small bee objects).
Reported as mean ± std across 5 fold models evaluated on the fixed test set.

| Model    | Test AP₅₀      | Test AP        | Test AR₁₀₀    | Val AP₅₀       | Val AP         |
|----------|---------------|---------------|---------------|---------------|---------------|
| YOLO11-m | **0.919 ± 0.010** | **0.516 ± 0.005** | **0.596 ± 0.006** | 0.962 ± 0.051 | 0.569 ± 0.051 |
| YOLO11-n | 0.918 ± 0.014 | 0.509 ± 0.016 | 0.590 ± 0.013 | 0.963 ± 0.045 | 0.565 ± 0.047 |
| YOLOv8-n | 0.916 ± 0.024 | 0.511 ± 0.023 | 0.590 ± 0.019 | 0.965 ± 0.042 | 0.567 ± 0.050 |
| YOLOv8-s | 0.912 ± 0.013 | 0.507 ± 0.019 | 0.587 ± 0.013 | 0.967 ± 0.044 | 0.567 ± 0.047 |
| YOLOv8-m | 0.912 ± 0.015 | 0.492 ± 0.016 | 0.576 ± 0.016 | 0.969 ± 0.036 | 0.570 ± 0.050 |
| YOLO11-s | 0.910 ± 0.017 | 0.505 ± 0.016 | 0.586 ± 0.012 | 0.964 ± 0.044 | 0.561 ± 0.047 |
| YOLO26-s | 0.910 ± 0.026 | 0.512 ± 0.015 | 0.588 ± 0.019 | 0.965 ± 0.043 | 0.571 ± 0.055 |
| YOLO26-n | 0.908 ± 0.020 | 0.506 ± 0.018 | 0.589 ± 0.012 | 0.966 ± 0.041 | 0.575 ± 0.047 |
| YOLO26-m | 0.905 ± 0.022 | 0.507 ± 0.023 | 0.585 ± 0.017 | 0.966 ± 0.044 | 0.567 ± 0.048 |
| MOG2     | 0.171 ± 0.000 | 0.039 ± 0.000 | 0.089 ± 0.000 | 0.258 ± 0.137 | 0.073 ± 0.042 |
| KNN      | 0.108 ± 0.003 | 0.023 ± 0.000 | 0.060 ± 0.001 | 0.267 ± 0.142 | 0.087 ± 0.063 |

## Notes
- AP_large = −1 for all models (no large bees in dataset) — omitted from table.
- Fold 4 has fewer val frames (264) due to data distribution constraints — acknowledged as a dataset limitation.
- MOG2/KNN warmup is unreliable for very short videos (< 30 frames).
- Results sorted by Test AP₅₀ descending within each model family; baselines last.
