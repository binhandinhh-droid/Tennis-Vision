import cv2
import numpy as np
import config


def line_intersection(p1, p2, p3, p4):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return (px, py)


def _weighted_avg_line(candidates):
    """Trung bình có trọng số (theo độ dài) của các line cùng hướng
    -> ổn định hơn khi có nhiễu, thay vì chỉ chọn 1 line dài nhất."""

    total_len = sum(ln for _, ln in candidates)
    lx1 = sum(l[0] * ln for l, ln in candidates) / total_len
    ly1 = sum(l[1] * ln for l, ln in candidates) / total_len
    lx2 = sum(l[2] * ln for l, ln in candidates) / total_len
    ly2 = sum(l[3] * ln for l, ln in candidates) / total_len
    return (lx1, ly1, lx2, ly2)


def refine_keypoint_by_lines(
    frame,
    pt,
    crop_half=config.REFINE_CROP_HALF,
    white_thresh=config.REFINE_WHITE_THRESH,
    max_shift=config.REFINE_MAX_SHIFT,
):
    h, w = frame.shape[:2]
    x, y = pt

    x0 = max(0, int(x - crop_half))
    y0 = max(0, int(y - crop_half))
    x1c = min(w, int(x + crop_half))
    y1c = min(h, int(y + crop_half))

    crop = frame[y0:y1c, x0:x1c]

    if crop.size == 0:
        return pt

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # --- Threshold thích nghi: lấy percentile sáng nhất trong vùng crop
    # thay vì ngưỡng cố định (ánh sáng sân có thể khác nhau giữa các trận)
    local_thresh = max(white_thresh, np.percentile(gray, 85))
    _, mask = cv2.threshold(gray, local_thresh, 255, cv2.THRESH_BINARY)

    # Đóng khoảng hở nhỏ do nhiễu hạt trên mặt sân (đất nện, cỏ...)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=10, minLineLength=10, maxLineGap=5,
    )

    if lines is None or len(lines) < 2:
        return pt

    lines = np.asarray(lines).reshape(-1, 4)

    horiz, vert = [], []

    for l in lines:
        lx1, ly1, lx2, ly2 = l
        length = np.hypot(lx2 - lx1, ly2 - ly1)
        angle = abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))

        if angle < 25 or angle > 155:
            horiz.append((l, length))
        elif 65 < angle < 115:
            vert.append((l, length))

    if not horiz or not vert:
        return pt

    hl = _weighted_avg_line(horiz)
    vl = _weighted_avg_line(vert)

    inter = line_intersection(
        (hl[0], hl[1]), (hl[2], hl[3]),
        (vl[0], vl[1]), (vl[2], vl[3]),
    )

    if inter is None:
        return pt

    ix, iy = inter

    if not (0 <= ix <= (x1c - x0) and 0 <= iy <= (y1c - y0)):
        return pt

    refined = (x0 + ix, y0 + iy)

    # --- Sanity check: nếu điểm refine di chuyển quá xa so với dự đoán
    # gốc của model, khả năng cao là bắt nhầm line -> giữ nguyên điểm gốc
    if np.hypot(refined[0] - x, refined[1] - y) > max_shift:
        return pt

    return refined
