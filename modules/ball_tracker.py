import cv2
import numpy as np
import config


class BallTracker:

    def __init__(self, ball_model,
                 search_margin=config.BALL_SEARCH_MARGIN,
                 local_conf=config.BALL_LOCAL_CONF,
                 global_conf=config.BALL_GLOBAL_CONF):

        self.ball_model = ball_model
        self.search_margin = search_margin
        self.local_conf = local_conf
        self.global_conf = global_conf

        self.last_ball_box = None

    def detect(self, frame, annotated_frame=None, debug=config.DEBUG_MODE):
        """
        Detect bóng trên frame hiện tại.

        Returns:
            center_point (cx, cy) hoặc None nếu không tìm thấy bóng.
        """

        height, width = frame.shape[:2]

        ball_results = None
        offset_x, offset_y = 0, 0

        # -------------------------------------------------
        # Local Search
        # -------------------------------------------------

        if self.last_ball_box is not None:

            lx1, ly1, lx2, ly2 = self.last_ball_box

            rx1 = max(0, lx1 - self.search_margin)
            ry1 = max(0, ly1 - self.search_margin)
            rx2 = min(width, lx2 + self.search_margin)
            ry2 = min(height, ly2 + self.search_margin)

            roi = frame[ry1:ry2, rx1:rx2]

            if debug and annotated_frame is not None:
                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

            if roi.size > 0:

                local_results = self.ball_model.predict(
                    roi, conf=self.local_conf, verbose=False
                )

                if len(local_results[0].boxes) > 0:
                    ball_results = local_results
                    offset_x, offset_y = rx1, ry1

        # -------------------------------------------------
        # Global Search
        # -------------------------------------------------

        if ball_results is None:

            global_results = self.ball_model.predict(
                frame, conf=self.global_conf, verbose=False
            )

            if len(global_results[0].boxes) > 0:
                ball_results = global_results
                offset_x, offset_y = 0, 0

        # -------------------------------------------------
        # Parse result
        # -------------------------------------------------

        if ball_results is None:
            self.last_ball_box = None
            return None

        boxes = ball_results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confs)

        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = confs[best_idx]

        x1, y1, x2, y2 = map(int, box)
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y

        self.last_ball_box = (x1, y1, x2, y2)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if annotated_frame is not None:
            self._draw(annotated_frame, x1, y1, x2, y2, cx, cy, conf, debug)

        return (cx, cy)

    @staticmethod
    def _draw(annotated_frame, x1, y1, x2, y2, cx, cy, conf, debug):

        cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 255), -1)

        if debug:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(
                annotated_frame,
                f"Ball {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
