import cv2
import numpy as np

import config
from court_detector import detect_court, compute_court_homography


def update_court_by_optical_flow(prev_gray, gray, keypoints):
    """
    Theo dõi 14 keypoint sân bằng Lucas-Kanade optical flow giữa 2 frame
    liên tiếp, để bù chuyển động camera mà không cần chạy lại model nặng.
    
    Returns:
        (new_keypoints, ok)
        ok=True  -> track ổn, new_keypoints đã cập nhật
        ok=False -> track bị mất quá nhiều điểm, nên detect lại ngay
    """

    pts = keypoints[:, :2].astype(np.float32).reshape(-1, 1, 2)

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None, **config.LK_PARAMS
    )

    if new_pts is None or status is None:
        return keypoints, False

    status = status.reshape(-1)

    valid_ratio = float(status.sum()) / len(status)

    if valid_ratio < config.COURT_TRACK_MIN_VALID_RATIO:
        return keypoints, False

    good_prev = pts.reshape(-1, 2)[status == 1]
    good_new = new_pts.reshape(-1, 2)[status == 1]

    # Cần tối thiểu vài điểm để ước lượng affine ổn định
    if len(good_prev) < 4:
        return keypoints, False

    M, inlier_mask = cv2.estimateAffinePartial2D(
        good_prev,
        good_new,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if M is None:
        return keypoints, False

    # Áp phép biến đổi camera chung này lên TOÀN BỘ 14 keypoint gốc
    # (không chỉ những điểm track được) -> mọi điểm di chuyển thống
    # nhất theo đúng chuyển động camera đã ước lượng.
    all_prev_xy = keypoints[:, :2].astype(np.float32)

    updated_xy = (M[:, :2] @ all_prev_xy.T).T + M[:, 2]

    updated = keypoints.copy()
    updated[:, 0] = updated_xy[:, 0]
    updated[:, 1] = updated_xy[:, 1]

    return updated, True


class CourtTracker:
    """
    Bọc toàn bộ state (cache + optical flow) cho việc theo dõi sân
    qua các frame, để main loop không phải quản lý biến toàn cục.
    """

    def __init__(self, court_model, mini_w, mini_h,
                 update_interval=config.COURT_UPDATE_INTERVAL,
                 device=config.DEVICE):

        self.court_model = court_model
        self.mini_w = mini_w
        self.mini_h = mini_h
        self.update_interval = update_interval
        self.device = device

        self.frame_idx = 0

        self.cached_court = None   # (box, keypoints, score)
        self.cached_H = None       # homography ứng với cached_court hiện tại

        self.prev_gray = None

    @property
    def box(self):
        return self.cached_court[0] if self.cached_court else None

    @property
    def keypoints(self):
        return self.cached_court[1] if self.cached_court else None

    @property
    def score(self):
        return self.cached_court[2] if self.cached_court else None

    @property
    def homography(self):
        return self.cached_H

    def update(self, frame):
        """
        Cập nhật trạng thái sân cho frame hiện tại: ưu tiên optical
        flow (rẻ, chạy mọi frame), chỉ chạy lại model nặng theo chu kỳ
        hoặc khi track bị mất.
        """

        self.frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        need_full_detect = (
            self.frame_idx % self.update_interval == 1
            or self.cached_court is None
        )

        # --- Optical flow: bù chuyển động camera mỗi frame (rẻ) ---
        if (not need_full_detect) and self.cached_court is not None and self.prev_gray is not None:

            tracked_kp, ok = update_court_by_optical_flow(
                self.prev_gray, gray, self.cached_court[1]
            )

            if ok:
                xy = tracked_kp[:, :2]

                new_box = np.array([
                    xy[:, 0].min(), xy[:, 1].min(),
                    xy[:, 0].max(), xy[:, 1].max(),
                ])

                self.cached_court = (new_box, tracked_kp, self.cached_court[2])

                self.cached_H = compute_court_homography(
                    tracked_kp, self.mini_w, self.mini_h
                )
            else:
                # Track bị mất (VD camera cắt cảnh đột ngột) -> detect
                # lại ngay lập tức, không đợi chu kỳ tiếp theo.
                need_full_detect = True

        # --- Model nặng: chỉ chạy định kỳ hoặc khi vừa mất track ---
        if need_full_detect:
            result = detect_court(frame, self.court_model, self.device)

            if result is not None:
                self.cached_court = result
                self.cached_H = compute_court_homography(
                    result[1], self.mini_w, self.mini_h
                )

        self.prev_gray = gray

        return self.cached_court, self.cached_H
