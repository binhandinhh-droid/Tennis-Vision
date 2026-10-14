"""
Measure the size of the crop fed to the person model to choose PERSON_ENGINE_IMGSZ (h, w).
Only runs the court model (not YOLO), and detects the court only ~1 time / second of video, so it is fairly fast.

    python measure_person_crop.py [video_path] [samples_per_second]

With an engine of width 1280, the person scale stays the same as a square 1280 engine when the engine height
>= 1280 * ch / max(cw, ch). Prints p95 / p99 / max of that value; choose p99 (or max if you want
absolutely no quality change).
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
        raise SystemExit(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / per_second)))

    model = load_court_model(COURT_MODEL_PATH)
    print(f"Video {total} frames @ {fps:.1f}fps, sampling every {step} frames (~{total // step} detections)")

    need, tried, idx, t0 = [], 0, 0, time.time()

    while cap.grab():                       # grab: skips the image conversion step, cheaper than read()
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
                    print(f"  frame {idx}/{total} | court seen {len(need)}/{tried} times | {time.time() - t0:.0f}s", flush=True)

        idx += 1

    cap.release()

    if not need:
        raise SystemExit("Could not detect the court in any sample.")

    need = np.array(need)
    p99 = int(np.percentile(need, 99))

    print(f"\n{len(need)}/{tried} samples with a court. Required engine height: "
          f"p50 {int(np.median(need))} | p95 {int(np.percentile(need, 95))} | p99 {p99} | max {int(need.max())}")
    print(f"\nRecommended in config.py:\n    PERSON_ENGINE_IMGSZ = ({p99}, {PERSON_IMGSZ})")
    print("Then run: python export_engines.py (EXPORT_PERSON = True, FORCE = True)")


if __name__ == "__main__":
    main(INPUT_VIDEO if len(sys.argv) < 2 else sys.argv[1],
         float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
