"""Track bóng: local search quanh vị trí cũ, fallback quét toàn frame."""

import cv2
import numpy as np

from config import (
    DEBUG_MODE, BALL_SEARCH_MARGIN, BALL_LOCAL_CONF, BALL_GLOBAL_CONF,
    BALL_LOST_PATIENCE, BALL_SEARCH_GROWTH,
    BALL_TRAIN_IMGSZ, BALL_LOCAL_IMGSZ_MIN, BALL_LOCAL_IMGSZ_MAX, BALL_LOCAL_MAX_MARGIN,
)


def _is_fixed_shape(model):
    """Engine TensorRT có shape cố định lúc export: không truyền imgsz lúc predict."""
    name = str(getattr(model, "model_name", "") or getattr(model, "ckpt_path", "") or "")
    return name.lower().endswith(".engine")


def _roi_imgsz(roi_w, roi_h, frame_w):
    """
    imgsz cho ROI để bóng có cùng kích thước pixel như lúc train:
    lúc train, ảnh full-frame bị resize với tỉ lệ BALL_TRAIN_IMGSZ / frame_w -> áp cùng tỉ lệ cho ROI.
    """
    scale = BALL_TRAIN_IMGSZ / float(frame_w)
    size = int(np.ceil(max(roi_w, roi_h) * scale / 32.0) * 32)
    return int(min(BALL_LOCAL_IMGSZ_MAX, max(BALL_LOCAL_IMGSZ_MIN, size)))


class BallTracker:

    def __init__(self, ball_model, local_model=None):

        self.ball_model = ball_model

        # Model nhỏ (engine imgsz thấp) chuyên cho vùng ROI; không có thì dùng chung model global.
        self.local_model = local_model if local_model is not None else ball_model

        # Engine shape cố định thì imgsz đã nằm trong lúc export; còn model .pt thì tính imgsz theo ROI.
        self._dynamic_imgsz = not _is_fixed_shape(self.local_model)

        # Giữ vùng tìm kiếm cũ vài frame sau khi mất bóng - CHỈ có ý nghĩa khi local
        # search rẻ (có local model riêng); dùng chung model thì patience = 0.
        self.lost_patience = BALL_LOST_PATIENCE if local_model is not None else 0
        self.lost_frames = 0

        self.last_ball_box = None

    def reset(self):
        """Chuyển cảnh: bỏ vùng tìm kiếm cũ."""
        self.last_ball_box = None
        self.lost_frames = 0

    def detect(self, frame, annotated_frame):
        """Returns: (cx, cy) hoặc None nếu không thấy bóng."""

        height, width = frame.shape[:2]

        result = None
        offset_x, offset_y = 0, 0

        # ---- Local search quanh vị trí cuối cùng thấy bóng ----
        if self.last_ball_box is not None:

            lx1, ly1, lx2, ly2 = self.last_ball_box

            # Mất bóng càng lâu thì vùng tìm càng rộng (có trần để bóng không bị co quá nhỏ khi resize)
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

        # ---- Không thấy bóng ----
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