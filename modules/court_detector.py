import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

import config
from geometry import refine_keypoint_by_lines


def detect_court(frame, court_model, device=config.DEVICE):
    """
    Chạy KeypointRCNN trên 1 frame để lấy 14 keypoint sân, sau đó
    refine từng điểm bằng CV classical (snap vào giao điểm 2 line trắng).

    Returns:
        (box, keypoints, score) hoặc None nếu không detect được / score thấp.
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).to(device)

    # Nếu model đang chạy ở half precision (đã quantize FP16), input
    # cũng phải cùng dtype, nếu không torch sẽ báo lỗi mismatch.
    model_dtype = next(court_model.parameters()).dtype
    tensor = tensor.to(model_dtype)

    with torch.no_grad():
        prediction = court_model([tensor])[0]

    if len(prediction["scores"]) == 0:
        return None

    best = prediction["scores"].argmax()
    score = prediction["scores"][best].item()

    if score < config.COURT_DETECT_SCORE_THRESH:
        return None

    # .float() trước khi sang numpy: nếu model chạy FP16, các bước xử
    # lý CV phía sau (refine_keypoint_by_lines, OpenCV...) cần FP32/64
    # ổn định hơn, tránh mất độ chính xác số học ở bước hình học.
    box = prediction["boxes"][best].float().cpu().numpy()
    keypoints = prediction["keypoints"][best].float().cpu().numpy()  # (14, 3)

    kp_xy = keypoints[:, :2].copy()

    # --- Classical CV local refinement (snap to line intersection) ---
    for i in range(len(kp_xy)):
        kp_xy[i] = refine_keypoint_by_lines(frame, kp_xy[i])

    keypoints[:, :2] = kp_xy

    return box, keypoints, score


def draw_court(img, box, keypoints, score, debug=config.DEBUG_MODE):
    """Vẽ box (nếu debug) + luôn đánh số thứ tự từng keypoint."""

    if debug:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        cv2.putText(
            img,
            f"Court {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

    for i, kp in enumerate(keypoints):
        x = int(kp[0])
        y = int(kp[1])

        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        cv2.putText(
            img,
            str(i),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )


def compute_court_homography(keypoints, mini_w, mini_h):
    """
    Tính ma trận homography ánh xạ từ 4 góc sân thật (trên frame)
    sang 4 góc của sân chuẩn hoá trên mini-map.

    Thứ tự keypoint thực tế của model: 0=TL, 1=TR, 2=BL, 3=BR.
    """

    src = np.array([
        keypoints[0][:2],  # top-left
        keypoints[1][:2],  # top-right
        keypoints[3][:2],  # bottom-right
        keypoints[2][:2],  # bottom-left
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [mini_w, 0],
        [mini_w, mini_h],
        [0, mini_h],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src, dst)


def project_point(H, pt):
    """Chiếu 1 điểm (x, y) trên frame gốc qua homography H."""

    p = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    res = H @ p

    if abs(res[2]) < 1e-6:
        return None

    return (res[0] / res[2], res[1] / res[2])
