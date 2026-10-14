"""Ball tracking: local search around the previous position, fallback to a full-frame scan."""

import cv2
import numpy as np

from config import (
    DEBUG_MODE, BALL_SEARCH_MARGIN, BALL_LOCAL_CONF, BALL_GLOBAL_CONF,
    BALL_LOST_PATIENCE, BALL_SEARCH_GROWTH,
    BALL_TRAIN_IMGSZ, BALL_LOCAL_IMGSZ_MIN, BALL_LOCAL_IMGSZ_MAX, BALL_LOCAL_MAX_MARGIN,
)


def _is_fixed_shape(model):
    """TensorRT engines have a fixed shape set at export time: do not pass imgsz at predict time."""
    name = str(getattr(model, "model_name", "") or getattr(model, "ckpt_path", "") or "")
    return name.lower().endswith(".engine")


def _roi_imgsz(roi_w, roi_h, frame_w):
    """
    imgsz for the ROI so the ball has the same pixel size as during training:
    at training time a full-frame image is resized by BALL_TRAIN_IMGSZ / frame_w -> apply the same ratio to the ROI.
    """
    scale = BALL_TRAIN_IMGSZ / float(frame_w)
    size = int(np.ceil(max(roi_w, roi_h) * scale / 32.0) * 32)
    return int(min(BALL_LOCAL_IMGSZ_MAX, max(BALL_LOCAL_IMGSZ_MIN, size)))


class BallTracker:

    def __init__(self, ball_model, local_model=None):

        self.ball_model = ball_model

        # Small model (low-imgsz engine) dedicated to the ROI; if absent, share the global model.
        self.local_model = local_model if local_model is not None else ball_model

        # A fixed-shape engine already has imgsz baked in at export; a .pt model computes imgsz from the ROI.
        self._dynamic_imgsz = not _is_fixed_shape(self.local_model)

        # Keep the old search region for a few frames after losing the ball - ONLY meaningful when the
        # local search is cheap (a dedicated local model); with a shared model patience = 0.
        self.lost_patience = BALL_LOST_PATIENCE if local_model is not None else 0
        self.lost_frames = 0

        self.last_ball_box = None

    def reset(self):
        """Scene change: drop the old search region."""
        self.last_ball_box = None
        self.lost_frames = 0

    def detect(self, frame, annotated_frame):
        """Returns: (cx, cy), or None if the ball is not seen."""

        height, width = frame.shape[:2]

        result = None
        offset_x, offset_y = 0, 0

        # ---- Local search around the last position where the ball was seen ----
        if self.last_ball_box is not None:

            lx1, ly1, lx2, ly2 = self.last_ball_box

            # The longer the ball is lost, the wider the search region (capped so the ball does not shrink too much when resized)
            margin = int(BALL_SEARCH_MARGIN * (1.0 + BALL_SEARCH_GROWTH * self.lost_frames))
            margin = min(margin, max(BALL_SEARCH_MARGIN, BALL_LOCAL_MAX_MARGIN))

            rx1 = max(0, lx1 - margin)
            ry1 = max(0, ly1 - margin)
            rx2 = min(width, lx2 + margin)
            ry2 = min(height, ly2 + margin)

            if DEBUG_MODE:
                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

            kwargs = {}

            if self._dynamic_imgsz:
                kwargs["imgsz"] = _roi_imgsz(rx2 - rx1, ry2 - ry1, width)

            local = self.local_model.predict(
                frame[ry1:ry2, rx1:rx2], conf=BALL_LOCAL_CONF, verbose=False, **kwargs
            )[0]

            if len(local.boxes) > 0:
                result = local
                offset_x, offset_y = rx1, ry1

        # ---- Global search ----
        if result is None:
            global_ = self.ball_model.predict(frame, conf=BALL_GLOBAL_CONF, verbose=False)[0]

            if len(global_.boxes) > 0:
                result = global_

        # ---- Ball not found ----
        if result is None:
            if self.last_ball_box is not None:
                self.lost_frames += 1

                if self.lost_frames > self.lost_patience:
                    self.last_ball_box = None
                    self.lost_frames = 0

            return None

        # ---- Parse result ----
        boxes = result.boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confs)

        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].cpu().numpy())
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y

        self.last_ball_box = (x1, y1, x2, y2)
        self.lost_frames = 0

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 255), -1)

        if DEBUG_MODE:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame, f"Ball {confs[best_idx]:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        return (cx, cy)
