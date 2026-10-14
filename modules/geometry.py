"""Geometry: line intersection, court keypoint refinement, frame -> mini-map homography."""

import cv2
import numpy as np

from config import (
    REFINE_CROP_HALF, REFINE_WHITE_THRESH, REFINE_MAX_SHIFT, _MINI_DST,
)


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
    """Length-weighted average of lines with the same orientation."""

    total_len = sum(ln for _, ln in candidates)
    return tuple(
        sum(l[k] * ln for l, ln in candidates) / total_len for k in range(4)
    )


def refine_keypoint_by_lines(frame, pt):
    """Snap a keypoint to the intersection of the 2 nearest white lines."""

    h, w = frame.shape[:2]
    x, y = pt

    x0 = max(0, int(x - REFINE_CROP_HALF))
    y0 = max(0, int(y - REFINE_CROP_HALF))
    x1c = min(w, int(x + REFINE_CROP_HALF))
    y1c = min(h, int(y + REFINE_CROP_HALF))

    crop = frame[y0:y1c, x0:x1c]

    if crop.size == 0:   # keypoint is outside the frame
        return pt

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold based on the brightest percentile inside the crop
    local_thresh = max(REFINE_WHITE_THRESH, np.percentile(gray, 85))
    _, mask = cv2.threshold(gray, local_thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=5,
    )

    if lines is None or len(lines) < 2:
        return pt

    horiz, vert = [], []

    for l in lines.reshape(-1, 4):
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

    inter = line_intersection((hl[0], hl[1]), (hl[2], hl[3]), (vl[0], vl[1]), (vl[2], vl[3]))

    if inter is None:
        return pt

    ix, iy = inter

    if not (0 <= ix <= (x1c - x0) and 0 <= iy <= (y1c - y0)):
        return pt

    refined = (x0 + ix, y0 + iy)

    # Shifted too far from the original prediction -> probably picked up the wrong line, keep the original point
    if np.hypot(refined[0] - x, refined[1] - y) > REFINE_MAX_SHIFT:
        return pt

    return refined


def compute_court_homography(keypoints):
    """Homography from the 4 court corners on the frame to the mini-map. Model: 0=TL, 1=TR, 2=BL, 3=BR."""

    src = np.array([
        keypoints[0][:2],  # top-left
        keypoints[1][:2],  # top-right
        keypoints[3][:2],  # bottom-right
        keypoints[2][:2],  # bottom-left
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src, _MINI_DST)


def project_point(H, pt):
    """Project a point (x, y) on the original frame through homography H."""

    res = H @ np.array([pt[0], pt[1], 1.0], dtype=np.float32)

    if abs(res[2]) < 1e-6:
        return None

    return (res[0] / res[2], res[1] / res[2])
