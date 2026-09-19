import cv2
import numpy as np

import config
from court_detector import project_point


def _draw_court_lines(canvas, mini_w, mini_h, pad):
    """Vẽ khung sân + biên đơn + vạch giao bóng + đường giữa + lưới."""

    # --- Khung sân (biên đôi) ---
    cv2.rectangle(canvas, (pad, pad), (pad + mini_w, pad + mini_h), (255, 255, 255), 2)

    # --- Biên đơn (2 đường dọc thu vào) ---
    inset = int(mini_w * config.SINGLES_INSET_RATIO)

    cv2.line(canvas, (pad + inset, pad), (pad + inset, pad + mini_h), (255, 255, 255), 1)
    cv2.line(canvas, (pad + mini_w - inset, pad), (pad + mini_w - inset, pad + mini_h), (255, 255, 255), 1)

    # --- Vạch giao bóng (service line) 2 bên ---
    near_y = int(mini_h * config.SERVICE_LINE_RATIO_NEAR)
    far_y = int(mini_h * config.SERVICE_LINE_RATIO_FAR)

    cv2.line(canvas, (pad + inset, pad + near_y), (pad + mini_w - inset, pad + near_y), (255, 255, 255), 1)
    cv2.line(canvas, (pad + inset, pad + far_y), (pad + mini_w - inset, pad + far_y), (255, 255, 255), 1)

    # --- Đường giữa (center line), nối 2 vạch giao bóng ---
    center_x = pad + mini_w // 2
    cv2.line(canvas, (center_x, pad + near_y), (center_x, pad + far_y), (255, 255, 255), 1)

    # --- Lưới (net), ở giữa sân theo chiều dài ---
    net_y = pad + mini_h // 2
    cv2.line(canvas, (pad, net_y), (pad + mini_w, net_y), (0, 255, 255), 2)


def _blit_canvas(frame, canvas, canvas_w, canvas_h):
    """Ghép canvas mini-map lên góc trên bên phải của frame chính."""

    fh, fw = frame.shape[:2]

    x_off = fw - canvas_w - config.MINI_MARGIN_RIGHT
    y_off = config.MINI_MARGIN_TOP

    if x_off < 0 or y_off < 0:
        return

    roi = frame[y_off:y_off + canvas_h, x_off:x_off + canvas_w]
    blended = cv2.addWeighted(roi, 0.25, canvas, 0.75, 0)
    frame[y_off:y_off + canvas_h, x_off:x_off + canvas_w] = blended

    cv2.rectangle(
        frame,
        (x_off, y_off),
        (x_off + canvas_w, y_off + canvas_h),
        (255, 255, 255),
        1,
    )


def draw_mini_court(frame, H, players, ball_point,
                     mini_w=config.MINI_COURT_W, mini_h=config.MINI_COURT_H,
                     pad=config.MINI_PADDING):
    """
    Vẽ mini-map sân tennis (top-down) ở góc trên bên phải của frame.

    Args:
        H:            homography đã tính sẵn cho frame hiện tại (dùng
                       chung với classify_player để đảm bảo nhất quán).
        players:      list các dict {"point": (x, y), "label": "Player 1"/"Player 2"}
                       (point là toạ độ điểm chân trên frame gốc)
        ball_point:   (x, y) toạ độ tâm bóng trên frame gốc, hoặc None
    """

    canvas_w = mini_w + pad * 2
    canvas_h = mini_h + pad * 2

    canvas = np.full((canvas_h, canvas_w, 3), (25, 25, 25), dtype=np.uint8)

    _draw_court_lines(canvas, mini_w, mini_h, pad)

    if H is None:
        fh, fw = frame.shape[:2]
        x_off = fw - canvas_w - config.MINI_MARGIN_RIGHT
        y_off = config.MINI_MARGIN_TOP

        if x_off < 0 or y_off < 0:
            return

        frame[y_off:y_off + canvas_h, x_off:x_off + canvas_w] = canvas
        return

    # --- Chiếu vị trí player + bóng lên mini-map ---
    for p in players:

        proj = project_point(H, p["point"])

        if proj is None:
            continue

        px, py = proj
        px = float(np.clip(px, 0, mini_w))
        py = float(np.clip(py, 0, mini_h))

        color = (0, 255, 0) if p["label"] == "Player 1" else (255, 128, 0)

        cv2.circle(canvas, (pad + int(px), pad + int(py)), 6, color, -1)
        cv2.circle(canvas, (pad + int(px), pad + int(py)), 6, (0, 0, 0), 1)

    if ball_point is not None:

        proj = project_point(H, ball_point)

        if proj is not None:
            bx, by = proj

            # chỉ vẽ nếu bóng nằm gần khu vực sân (kể cả hơi bay ra ngoài
            # biên một chút khi giao bóng / trả bóng ở góc)
            if -30 <= bx <= mini_w + 30 and -30 <= by <= mini_h + 30:
                bx = float(np.clip(bx, 0, mini_w))
                by = float(np.clip(by, 0, mini_h))

                cv2.circle(canvas, (pad + int(bx), pad + int(by)), 4, (0, 0, 255), -1)

    _blit_canvas(frame, canvas, canvas_w, canvas_h)
