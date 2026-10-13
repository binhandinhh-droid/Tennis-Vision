"""
Đo kích thước vùng crop đưa vào model người để chọn PERSON_ENGINE_IMGSZ (h, w).
Chỉ chạy model sân (không chạy YOLO), và chỉ detect sân ~1 lần / giây video nên khá nhanh.

    python measure_person_crop.py [đường_dẫn_video] [số_lần_lấy_mẫu_mỗi_giây]

Với engine rộng 1280, scale của người giữ nguyên như engine vuông 1280 khi chiều cao engine
>= 1280 * ch / max(cw, ch). In ra p95 / p99 / max của giá trị đó; chọn p99 (hoặc max nếu muốn
tuyệt đối không đổi chất lượng).
"""

import math
import sys
import time

import cv2
import numpy as np

from config import INPUT_VIDEO, COURT_MODEL_PATH, PERSON_IMGSZ
from models import load_court_model
from court_detector import detect_court
from player_tracker import _crop_around_court


def main(video, per_second=1.0):
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / per_second)))

    model = load_court_model(COURT_MODEL_PATH)
    print(f"Video {total} frame @ {fps:.1f}fps, lấy mẫu mỗi {step} frame (~{total // step} lần detect)")

    need, tried, idx, t0 = [], 0, 0, time.time()

    while cap.grab():                       # grab: bỏ qua bước chuyển sang ảnh, rẻ hơn read()
        if idx % step == 0:
            ok, frame = cap.retrieve()

            if ok:
                tried += 1
                court = detect_court(frame, model)

                if court is not None:
                    src, _, _ = _crop_around_court(frame, court[0])
                    ch, cw = src.shape[:2]
                    need.append(math.ceil(PERSON_IMGSZ * ch / max(cw, ch) / 32.0) * 32)

                if tried % 10 == 0:
                    print(f"  frame {idx}/{total} | {len(need)}/{tried} lần thấy sân | {time.time() - t0:.0f}s", flush=True)

        idx += 1

    cap.release()

    if not need:
        raise SystemExit("Không detect được sân ở mẫu nào.")

    need = np.array(need)
    p99 = int(np.percentile(need, 99))

    print(f"\n{len(need)}/{tried} mẫu có sân. Chiều cao engine cần: "
          f"p50 {int(np.median(need))} | p95 {int(np.percentile(need, 95))} | p99 {p99} | max {int(need.max())}")
    print(f"\nKhuyến nghị trong config.py:\n    PERSON_ENGINE_IMGSZ = ({p99}, {PERSON_IMGSZ})")
    print("Rồi chạy: python export_engines.py (EXPORT_PERSON = True, FORCE = True)")


if __name__ == "__main__":
    main(INPUT_VIDEO if len(sys.argv) < 2 else sys.argv[1],
         float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)