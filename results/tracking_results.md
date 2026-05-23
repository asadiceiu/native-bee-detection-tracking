# Tracking Benchmark Results

Mean ± std across 13 sequences (IoU threshold 0.5 for matching).

## YOLO Detections (end-to-end)

| Tracker      | Mode   | Precision     | Recall        | F1            | MOTA          | IDSW | IDF1          |
| ------------ | ------ | ------------- | ------------- | ------------- | ------------- | ---- | ------------- |
| Kalman (IoU) | YOLO   | 0.849 ± 0.111 | 0.992 ± 0.011 | 0.911 ± 0.066 | 0.753 ± 0.195 | 281  | 0.759 ± 0.153 |
| ByteTrack    | YOLO   | 0.892 ± 0.102 | 0.814 ± 0.185 | 0.839 ± 0.131 | 0.700 ± 0.231 | 64   | 0.779 ± 0.175 |
| BoT-SORT     | YOLO   | 0.895 ± 0.101 | 0.813 ± 0.184 | 0.841 ± 0.131 | 0.700 ± 0.231 | 79   | 0.721 ± 0.186 |

## GT Oracle (tracker quality only)

| Tracker      | Mode   | Precision     | Recall        | F1            | MOTA          | IDSW | IDF1          |
| ------------ | ------ | ------------- | ------------- | ------------- | ------------- | ---- | ------------- |
| Kalman (IoU) | Oracle | 0.998 ± 0.004 | 0.998 ± 0.004 | 0.998 ± 0.004 | 0.958 ± 0.077 | 275  | 0.875 ± 0.127 |
| ByteTrack    | Oracle | 1.000 ± 0.000 | 0.929 ± 0.094 | 0.960 ± 0.057 | 0.928 ± 0.096 | 4    | 0.926 ± 0.087 |
| BoT-SORT     | Oracle | 1.000 ± 0.000 | 0.929 ± 0.094 | 0.960 ± 0.057 | 0.928 ± 0.096 | 4    | 0.926 ± 0.087 |

*3 trackers × 2 modes × 13 sequences*
