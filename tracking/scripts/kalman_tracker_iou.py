#!/usr/bin/env python3
"""
IoU-based Kalman tracker for benchmark evaluation.

Implements the SORT algorithm:
  - 8D constant-velocity state: [cx, cy, w, h, vx, vy, vw, vh]
  - Hungarian algorithm on 1 − IoU cost matrix
  - IoU threshold for association
  - Max age before track deletion
  - Min hits before track is reported (to suppress false positives)

Does NOT modify or import the existing kalman_tracker.py.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


def box_xyxy_to_state(box: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] → [cx, cy, w, h, 0, 0, 0, 0]."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0])


def state_to_xyxy(state: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h, ...] → [x1, y1, x2, y2]."""
    cx, cy, w, h = state[0], state[1], state[2], state[3]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _make_kalman(initial_box: np.ndarray) -> KalmanFilter:
    kf = KalmanFilter(dim_x=8, dim_z=4)
    # State transition: constant velocity
    kf.F = np.eye(8)
    kf.F[:4, 4:] = np.eye(4)
    # Observation: observe [cx, cy, w, h]
    kf.H = np.eye(4, 8)
    # Measurement noise
    kf.R *= 10.0
    kf.R[2:, 2:] *= 10.0
    # Process noise
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    # Initial state covariance
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    kf.x[:4] = box_xyxy_to_state(initial_box)[:4].reshape(4, 1)
    return kf


class Track:
    _id_counter = 0

    def __init__(self, box: np.ndarray):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.kf = _make_kalman(box)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, box: np.ndarray):
        z = box_xyxy_to_state(box)[:4].reshape(4, 1)
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0

    def get_box(self) -> np.ndarray:
        return state_to_xyxy(self.kf.x[:4, 0])

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits


class KalmanTrackerIoU:
    """
    SORT-style IoU-based Kalman tracker.

    Parameters
    ----------
    max_age:     frames a track survives without a detection match
    min_hits:    detections needed before a track is reported
    iou_thresh:  minimum IoU for a detection-track match
    """

    def __init__(self, max_age: int = 10, min_hits: int = 1, iou_thresh: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.tracks: list[Track] = []
        Track._id_counter = 0

    def reset(self):
        self.tracks = []
        Track._id_counter = 0

    def update(self, detections: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """
        Update tracker with detections for one frame.

        Parameters
        ----------
        detections: (N, 4) array of [x1, y1, x2, y2] boxes (or empty)

        Returns
        -------
        List of (track_id, [x1, y1, x2, y2]) for confirmed active tracks.
        """
        for trk in self.tracks:
            trk.predict()

        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._active_tracks()

        predicted_boxes = np.array([t.get_box() for t in self.tracks])

        if len(self.tracks) > 0:
            iou = iou_batch(predicted_boxes, detections)   # (n_tracks, n_dets)
            cost = 1.0 - iou
            trk_idx, det_idx = linear_sum_assignment(cost)
            matched_trk = set()
            matched_det = set()
            for ti, di in zip(trk_idx, det_idx):
                if iou[ti, di] >= self.iou_thresh:
                    self.tracks[ti].update(detections[di])
                    matched_trk.add(ti)
                    matched_det.add(di)
        else:
            matched_trk, matched_det = set(), set()

        # Initialise new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det:
                self.tracks.append(Track(det))

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self._active_tracks()

    def _active_tracks(self) -> list[tuple[int, np.ndarray]]:
        return [
            (t.track_id, t.get_box())
            for t in self.tracks
            if t.time_since_update == 0 and t.is_confirmed(self.min_hits)
        ]
