"""Detect sân bằng KeypointRCNN + refine bằng CV cổ điển + kiểm tra kết quả có đúng là sân."""

import cv2
import numpy as np
from torchvision.transforms import functional as F
import torch

from config import (
    DEVICE, DEBUG_MODE, COURT_DETECT_SCORE_THRESH,
    COURT_MIN_AREA_RATIO, COURT_MAX_AREA_RATIO, COURT_INSIDE_TOL, COURT_INSIDE_MIN_RATIO,
    COURT_EDGE_CONTRAST, COURT_EDGE_MIN_FRAC, COURT_EDGE_MIN_GOOD,
)
from geometry import refine_keypoint_by_lines


def _fail(reason):
    if DEBUG_MODE:
        print(f"[Court] bỏ detect: {reason}")
    return False


def _edge_is_line(gray, a, b):
    """
    Cạnh a->b có "dạng vạch trắng mảnh" không: điểm mẫu trên cạnh sáng hơn 2 bên (±8px).
    Returns (n_valid, frac_line); n_valid = số điểm mẫu nằm trong khung hình.
    """

    h, w = gray.shape
    d = b - a
    length = float(np.hypot(*d))

    if length < 20:
        return 0, 0.0

    n = np.array([-d[1], d[0]]) / length   # pháp tuyến đơn vị
    t = np.linspace(0.1, 0.9, 24)[:, None]
    pts = a + t * d

    def sample(offset):
        q = np.rint(pts + offset * n).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        val = np.full(len(q), np.nan, dtype=np.float32)
        val[ok] = gray[q[ok, 1], q[ok, 0]]
        return val

    with np.errstate(invalid="ignore"):
        center = np.nanmax(np.stack([sample(k) for k in (-3, -2, -1, 0, 1, 2, 3)]), axis=0) \
            if not np.all(np.isnan(sample(0))) else np.full(len(pts), np.nan)
        side = (sample(-8) + sample(8)) / 2.0
        valid = ~np.isnan(center) & ~np.isnan(side)

        if valid.sum() == 0:
            return 0, 0.0

        line = (center[valid] - side[valid]) >= COURT_EDGE_CONTRAST

    return int(valid.sum()), float(line.mean())


def is_valid_court(frame, keypoints):
    """
    Bộ 14 keypoint có thật sự là sân tennis không (KeypointRCNN hay "thấy" sân trên mặt người,
    khán đài...). Kiểm tra độc lập với score:
      1. 4 góc ngoài (0=TL, 1=TR, 2=BL, 3=BR) tạo tứ giác lồi, đúng thứ tự, diện tích hợp lý
      2. cạnh xa (trên) không rộng hơn cạnh gần (dưới) - camera truyền hình nhìn từ sau baseline
      3. các keypoint còn lại nằm trong tứ giác 4 góc
      4. các cạnh tứ giác trùng với vạch trắng mảnh trên ảnh
    """

    h, w = frame.shape[:2]
    xy = keypoints[:, :2].astype(np.float64)
    q = np.array([xy[0], xy[1], xy[3], xy[2]])   # TL, TR, BR, BL

    # 1. lồi + đúng thứ tự
    cross = []

    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))

    if not all(v > 0 for v in cross):
        return _fail("4 góc không tạo tứ giác lồi đúng thứ tự")

    area = 0.5 * abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4)))
    ratio = area / float(w * h)

    if not (COURT_MIN_AREA_RATIO <= ratio <= COURT_MAX_AREA_RATIO):
        return _fail(f"diện tích tứ giác sân {ratio:.3f} x frame")

    # 2. cạnh xa không rộng hơn cạnh gần
    top_w = np.linalg.norm(q[1] - q[0])
    bot_w = np.linalg.norm(q[2] - q[3])

    if top_w > bot_w * 1.2:
        return _fail("cạnh xa rộng hơn cạnh gần")

    # 3. keypoint còn lại nằm trong tứ giác
    contour = q.astype(np.float32).reshape(-1, 1, 2)
    tol = COURT_INSIDE_TOL * np.sqrt(area)
    inside = [cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True) >= -tol for p in xy[4:]]

    if np.mean(inside) < COURT_INSIDE_MIN_RATIO:
        return _fail(f"chỉ {np.mean(inside):.0%} keypoint nằm trong tứ giác sân")

    # 4. cạnh tứ giác trùng vạch trắng
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    checked, good = 0, 0

    for i in range(4):
        n_valid, frac = _edge_is_line(gray, q[i], q[(i + 1) % 4])

        if n_valid >= 8:
            checked += 1
            good += frac >= COURT_EDGE_MIN_FRAC

    if checked >= 2 and good < min(COURT_EDGE_MIN_GOOD, checked):
        return _fail(f"chỉ {good}/{checked} cạnh nhìn ra vạch trắng")

    return True


def detect_court(frame, court_model, min_score=COURT_DETECT_SCORE_THRESH):
    """
    KeypointRCNN lấy 14 keypoint sân, refine từng điểm bằng CV cổ điển, rồi kiểm tra có đúng là sân.

    Returns:
        (box, keypoints, score) hoặc None.
    """

    tensor = F.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.to(DEVICE, next(court_model.parameters()).dtype)   # model FP16 -> input FP16

    with torch.no_grad():
        prediction = court_model([tensor])[0]

    scores = prediction["scores"]

    if len(scores) == 0:
        return None

    best = scores.argmax()
    score = scores[best].item()

    if score < min_score:
        return None

    box = prediction["boxes"][best].float().cpu().numpy()
    keypoints = prediction["keypoints"][best].float().cpu().numpy()  # (14, 3)

    for i in range(len(keypoints)):
        keypoints[i, :2] = refine_keypoint_by_lines(frame, keypoints[i, :2])

    if not is_valid_court(frame, keypoints):
        return None

    return box, keypoints, score


def draw_court(img, court):
    """Vẽ box (nếu debug) + luôn đánh số thứ tự từng keypoint."""

    box, keypoints, score = court

    if DEBUG_MODE:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            img, f"Court {score:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2,
        )

    for i, kp in enumerate(keypoints):
        x, y = int(kp[0]), int(kp[1])

        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(
            img, str(i), (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )
