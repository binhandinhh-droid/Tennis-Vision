"""Court tracker: optical flow mỗi frame + detect lại (KeypointRCNN) chạy nền khi cần."""

import threading

import cv2
import numpy as np
import torch

from config import (
    DEVICE, DEBUG_MODE, OPTICAL_FLOW_SCALE, LK_PARAMS,
    COURT_MIN_INTERVAL, COURT_MAX_INTERVAL, COURT_REDETECT_TRAVEL_PX,
    COURT_FLOW_MIN_INLIER_RATIO, COURT_FLOW_MAX_RESIDUAL_PX, COURT_FLOW_DEADBAND_PX,
    COURT_TRACK_MIN_VALID_RATIO, COURT_LOST_RETRY_INTERVAL, COURT_DROP_AFTER_FAILS,
    COURT_ACQUIRE_SCORE,
)
from court_detector import detect_court
from geometry import compute_court_homography


def _affine_to_3x3(M):
    A = np.eye(3)
    A[:2, :] = M
    return A


def _warp_points(A, xy):
    return (A[:2, :2] @ xy.T).T + A[:2, 2]


def _box_from_keypoints(kp):
    xy = kp[:, :2]
    return np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])


def track_court_by_optical_flow(ref_gray, gray, keypoints):
    """
    Lucas-Kanade optical flow cho 14 keypoint từ ảnh tham chiếu `ref_gray` (frame mà
    keypoints đang khớp) tới frame hiện tại. Ảnh xám đã thu nhỏ theo OPTICAL_FLOW_SCALE;
    toạ độ trả về ở hệ full-res.

    Returns:
        None nếu mất track, ngược lại (new_keypoints, M, motion, fit_ok)
        M      = affine 2x3 (full-res) mô tả chuyển động camera
        motion = độ dịch trung bình (px) của 14 keypoint
        fit_ok = False nếu phép biến đổi khớp kém (zoom / đổi góc / nhiều điểm lệch)
    """

    xy = keypoints[:, :2].astype(np.float32)
    pts = (xy * OPTICAL_FLOW_SCALE).reshape(-1, 1, 2)

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(ref_gray, gray, pts, None, **LK_PARAMS)

    good = status.reshape(-1) == 1

    if good.mean() < COURT_TRACK_MIN_VALID_RATIO:
        return None

    new_xy = new_pts.reshape(-1, 2)[good] / OPTICAL_FLOW_SCALE

    M, inliers = cv2.estimateAffinePartial2D(
        xy[good], new_xy, method=cv2.RANSAC, ransacReprojThreshold=5.0,
    )

    if M is None:
        return None

    inliers = inliers.ravel().astype(bool)
    residual = np.linalg.norm(_warp_points(M, xy[good]) - new_xy, axis=1)[inliers].mean()
    fit_ok = (inliers.mean() >= COURT_FLOW_MIN_INLIER_RATIO
              and residual <= COURT_FLOW_MAX_RESIDUAL_PX)

    # Áp phép biến đổi camera chung lên TOÀN BỘ 14 keypoint
    warped = _warp_points(M, xy)
    updated = keypoints.copy()
    updated[:, :2] = warped

    motion = float(np.linalg.norm(warped - xy, axis=1).mean())

    return updated, M, motion, fit_ok


_COURT_STREAM = None


def _get_court_stream():
    """Dùng lại 1 CUDA stream cho mọi job (tạo stream mới mỗi lần làm allocator của
    PyTorch không tái sử dụng được vùng nhớ -> cudaMalloc/cudaFree lặp lại gây đứng hình)."""
    global _COURT_STREAM
    if _COURT_STREAM is None:
        _COURT_STREAM = torch.cuda.Stream(device=DEVICE)
    return _COURT_STREAM


class _DetectJob:
    """
    Chạy detect_court trên thread nền (CUDA stream riêng) để main loop không khựng.

    `acc` là phép biến đổi camera TÍCH LŨY từ frame lúc submit tới frame hiện tại,
    dùng để đưa kết quả (ở hệ toạ độ frame cũ) về frame hiện tại.
    """

    def __init__(self, court_model, frame):
        self.acc = np.eye(3)
        self.result = None
        self.error = None
        self.done = threading.Event()
        threading.Thread(target=self._run, args=(court_model, frame), daemon=True).start()

    def _run(self, court_model, frame):
        try:
            if DEVICE.type == "cuda":
                stream = _get_court_stream()
                with torch.cuda.stream(stream):
                    self.result = detect_court(frame, court_model)
                stream.synchronize()
            else:
                self.result = detect_court(frame, court_model)
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            self.done.set()


class CourtTracker:
    """
    - Mỗi frame: optical flow (so với ảnh tham chiếu) bù chuyển động camera. Camera gần
      như đứng yên thì giữ nguyên sân + ảnh tham chiếu, nên không tích nhiễu và H ổn định.
    - Detect lại (KeypointRCNN, chạy NỀN) CHỈ khi cần, xem `_redetect_reason`:
        poor-fit      optical flow khớp kém (zoom / đổi góc / điểm lệch nhiều)
        camera-moved  keypoint đã trôi theo camera đủ xa kể từ lần detect cuối
        max-interval  lâu quá không detect (chốt chặn an toàn)
      Luôn cách nhau tối thiểu COURT_MIN_INTERVAL frame.
    - Chưa có sân / mất track: detect ĐỒNG BỘ ngay.
    """

    def __init__(self, court_model):
        self.court_model = court_model
        self.frame_idx = 0

        self.cached_court = None   # (box, keypoints, score)
        self.cached_H = None       # homography ứng với cached_court
        self.ref_gray = None       # ảnh xám (đã thu nhỏ) mà cached keypoints đang khớp

        self.last_detect_idx = -COURT_LOST_RETRY_INTERVAL   # frame_idx của lần detect gần nhất
        self.travel = 0.0          # tổng quãng đường keypoint trôi kể từ lần detect gần nhất
        self.poor_fit = False      # optical flow khớp kém kể từ lần detect gần nhất

        self._job = None
        self.sync_fail = 0         # số lần detect đồng bộ thất bại liên tiếp

    def _set_court(self, box, keypoints, score):
        self.cached_court = (box, keypoints, score)
        self.cached_H = compute_court_homography(keypoints)

    def _mark_detect(self):
        self.last_detect_idx = self.frame_idx
        self.travel = 0.0
        self.poor_fit = False

    def _redetect_reason(self):
        since = self.frame_idx - self.last_detect_idx

        if since < COURT_MIN_INTERVAL:
            return ""
        if self.poor_fit:
            return "poor-fit"
        if self.travel >= COURT_REDETECT_TRAVEL_PX:
            return "camera-moved"
        if since >= COURT_MAX_INTERVAL:
            return "max-interval"
        return ""

    def _drop_job(self):
        """Đợi job nền (nếu có) chạy xong rồi bỏ kết quả."""
        if self._job is not None:
            self._job.done.wait()
            self._job = None

    def _finish_job_if_ready(self):
        job = self._job

        if job is None or not job.done.is_set():
            return

        self._job = None

        if job.error is not None:
            print(f"[Court] detect nền bị lỗi: {job.error}")
            return

        if job.result is None:
            return

        _, kp, score = job.result

        # Kết quả ở hệ toạ độ frame lúc submit -> bù chuyển động camera tích luỹ
        kp = kp.copy()
        kp[:, :2] = _warp_points(job.acc, kp[:, :2])

        if DEBUG_MODE:   # độ lệch giữa sân đang track và sân detect lại: dùng để chỉnh ngưỡng
            gap = np.linalg.norm(kp[:, :2] - self.cached_court[1][:, :2], axis=1).mean()
            print(f"[Court] frame {self.frame_idx}: lệch so với optical flow {gap:.2f}px")

        self._set_court(_box_from_keypoints(kp), kp, score)

    def update(self, frame, gray=None):
        """Cập nhật sân cho frame hiện tại. Returns (cached_court, cached_H).
        gray: ảnh xám đã thu nhỏ theo OPTICAL_FLOW_SCALE (tính sẵn ở thread đọc); None thì tự tính."""

        self.frame_idx += 1

        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if OPTICAL_FLOW_SCALE != 1.0:
                gray = cv2.resize(
                    gray, None, fx=OPTICAL_FLOW_SCALE, fy=OPTICAL_FLOW_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

        need_sync = self.cached_court is None

        if need_sync and self.frame_idx - self.last_detect_idx < COURT_LOST_RETRY_INTERVAL:
            self.ref_gray = gray
            return None, None   # chưa có sân (cảnh không phải sân): không detect mỗi frame

        if not need_sync:
            tracked = track_court_by_optical_flow(self.ref_gray, gray, self.cached_court[1])

            if tracked is None:
                need_sync = True   # mất track (cắt cảnh...) -> detect lại ngay
            else:
                tracked_kp, M, motion, fit_ok = tracked

                # Camera đứng yên (dịch < deadband): giữ nguyên sân và ảnh tham chiếu
                if motion >= COURT_FLOW_DEADBAND_PX:
                    self._set_court(
                        _box_from_keypoints(tracked_kp), tracked_kp, self.cached_court[2]
                    )
                    self.ref_gray = gray
                    self.travel += motion
                    self.poor_fit = self.poor_fit or not fit_ok

                    # Cộng dồn chuyển động camera cho job nền đang chạy
                    if self._job is not None:
                        self._job.acc = _affine_to_3x3(M) @ self._job.acc

        if need_sync:
            self._drop_job()

            result = detect_court(frame, self.court_model, min_score=COURT_ACQUIRE_SCORE)

            if result is not None:
                self._set_court(*result)
                self.sync_fail = 0
            else:
                # Detect thất bại (độ tin cậy quá thấp / không thấy sân): đủ N lần thì bỏ sân cũ
                # thay vì giữ mãi sân của cảnh trước
                self.sync_fail += 1

                if self.sync_fail >= COURT_DROP_AFTER_FAILS:
                    self.cached_court = None
                    self.cached_H = None

            self.ref_gray = gray
            self._mark_detect()
        else:
            # Nhận kết quả detect nền (nếu đã xong), rồi mới quyết định submit job mới
            self._finish_job_if_ready()

            reason = self._redetect_reason()

            if reason and self._job is None:
                if DEBUG_MODE:
                    print(f"[Court] frame {self.frame_idx}: detect lại ({reason})")
                self._job = _DetectJob(self.court_model, frame)
                self._mark_detect()

        return self.cached_court, self.cached_H

    def reset(self):
        """Chuyển cảnh: bỏ sân cũ, detect lại từ đầu (nếu cảnh mới không có sân thì court = None)."""
        self._drop_job()
        self.cached_court = None
        self.cached_H = None
        self.ref_gray = None
        self.travel = 0.0
        self.poor_fit = False
        self.sync_fail = 0

    def close(self):
        self._drop_job()
