"""Court tracker: optical flow every frame + background re-detection (KeypointRCNN) when needed."""

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
    Lucas-Kanade optical flow for the 14 keypoints from the reference image `ref_gray` (the frame the
    keypoints currently match) to the current frame. Grayscale images are downscaled by OPTICAL_FLOW_SCALE;
    returned coordinates are in the full-resolution system.

    Returns:
        None if tracking is lost, otherwise (new_keypoints, M, motion, fit_ok)
        M      = affine 2x3 (full-res) describing the camera motion
        motion = mean displacement (px) of the 14 keypoints
        fit_ok = False if the transform fits poorly (zoom / angle change / many outlier points)
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

    # Apply the shared camera transform to ALL 14 keypoints
    warped = _warp_points(M, xy)
    updated = keypoints.copy()
    updated[:, :2] = warped

    motion = float(np.linalg.norm(warped - xy, axis=1).mean())

    return updated, M, motion, fit_ok


_COURT_STREAM = None


def _get_court_stream():
    """Reuse a single CUDA stream for every job (creating a new stream each time prevents PyTorch's
    allocator from reusing memory -> repeated cudaMalloc/cudaFree causes stutter)."""
    global _COURT_STREAM
    if _COURT_STREAM is None:
        _COURT_STREAM = torch.cuda.Stream(device=DEVICE)
    return _COURT_STREAM


class _DetectJob:
    """
    Run detect_court on a background thread (own CUDA stream) so the main loop does not stall.

    `acc` is the ACCUMULATED camera transform from the frame at submit time to the current frame,
    used to bring the result (in the old frame's coordinates) to the current frame.
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
    - Every frame: optical flow (against the reference image) compensates camera motion. When the camera is
      almost still, the court + reference image are kept as is, so noise does not accumulate and H stays stable.
    - Re-detection (KeypointRCNN, running in the BACKGROUND) ONLY when needed, see `_redetect_reason`:
        poor-fit      optical flow fits poorly (zoom / angle change / many outlier points)
        camera-moved  keypoints drifted with the camera far enough since the last detection
        max-interval  too long since the last detection (safety stop)
      Always at least COURT_MIN_INTERVAL frames apart.
    - No court yet / track lost: detect SYNCHRONOUSLY right away.
    """

    def __init__(self, court_model):
        self.court_model = court_model
        self.frame_idx = 0

        self.cached_court = None   # (box, keypoints, score)
        self.cached_H = None       # homography matching cached_court
        self.ref_gray = None       # (downscaled) grayscale image the cached keypoints currently match

        self.last_detect_idx = -COURT_LOST_RETRY_INTERVAL   # frame_idx of the most recent detection
        self.travel = 0.0          # total distance keypoints drifted since the most recent detection
        self.poor_fit = False      # optical flow fit poorly since the most recent detection

        self._job = None
        self.sync_fail = 0         # number of consecutive synchronous detection failures

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
        """Wait for the background job (if any) to finish, then discard its result."""
        if self._job is not None:
            self._job.done.wait()
            self._job = None

    def _finish_job_if_ready(self):
        job = self._job

        if job is None or not job.done.is_set():
            return

        self._job = None

        if job.error is not None:
            print(f"[Court] background detection failed: {job.error}")
            return

        if job.result is None:
            return

        _, kp, score = job.result

        # Result is in the coordinates of the frame at submit time -> compensate accumulated camera motion
        kp = kp.copy()
        kp[:, :2] = _warp_points(job.acc, kp[:, :2])

        if DEBUG_MODE:   # gap between the tracked court and the re-detected court: used to tune thresholds
            gap = np.linalg.norm(kp[:, :2] - self.cached_court[1][:, :2], axis=1).mean()
            print(f"[Court] frame {self.frame_idx}: offset vs optical flow {gap:.2f}px")

        self._set_court(_box_from_keypoints(kp), kp, score)

    def update(self, frame, gray=None):
        """Update the court for the current frame. Returns (cached_court, cached_H).
        gray: grayscale image already downscaled by OPTICAL_FLOW_SCALE (precomputed in the reader thread); computed here if None."""

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
            return None, None   # no court yet (non-court scene): do not detect every frame

        if not need_sync:
            tracked = track_court_by_optical_flow(self.ref_gray, gray, self.cached_court[1])

            if tracked is None:
                need_sync = True   # track lost (scene cut...) -> re-detect immediately
            else:
                tracked_kp, M, motion, fit_ok = tracked

                # Still camera (shift < deadband): keep the court and reference image as is
                if motion >= COURT_FLOW_DEADBAND_PX:
                    self._set_court(
                        _box_from_keypoints(tracked_kp), tracked_kp, self.cached_court[2]
                    )
                    self.ref_gray = gray
                    self.travel += motion
                    self.poor_fit = self.poor_fit or not fit_ok

                    # Accumulate camera motion for the running background job
                    if self._job is not None:
                        self._job.acc = _affine_to_3x3(M) @ self._job.acc

        if need_sync:
            self._drop_job()

            result = detect_court(frame, self.court_model, min_score=COURT_ACQUIRE_SCORE)

            if result is not None:
                self._set_court(*result)
                self.sync_fail = 0
            else:
                # Detection failed (confidence too low / no court seen): after N failures drop the old court
                # instead of keeping the previous scene's court forever
                self.sync_fail += 1

                if self.sync_fail >= COURT_DROP_AFTER_FAILS:
                    self.cached_court = None
                    self.cached_H = None

            self.ref_gray = gray
            self._mark_detect()
        else:
            # Collect the background detection result (if finished), then decide whether to submit a new job
            self._finish_job_if_ready()

            reason = self._redetect_reason()

            if reason and self._job is None:
                if DEBUG_MODE:
                    print(f"[Court] frame {self.frame_idx}: re-detecting ({reason})")
                self._job = _DetectJob(self.court_model, frame)
                self._mark_detect()

        return self.cached_court, self.cached_H

    def reset(self):
        """Scene change: drop the old court and detect from scratch (if the new scene has no court, court = None)."""
        self._drop_job()
        self.cached_court = None
        self.cached_H = None
        self.ref_gray = None
        self.travel = 0.0
        self.poor_fit = False
        self.sync_fail = 0

    def close(self):
        self._drop_job()
