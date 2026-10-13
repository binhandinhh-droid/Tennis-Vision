"""
Tennis Analysis - Inference

Pipeline: config -> models -> geometry -> court detector/tracker -> player tracker
-> ball tracker -> bounce detector -> mini-map -> main loop.

Chuyển cảnh (cut / dissolve) được dùng để reset court tracker, ball tracker và bounce detector.

Chạy:
    python main.py

Phím trong cửa sổ preview:  q thoát
"""

import argparse
import os
import time

import cv2

from config import INPUT_VIDEO, OUTPUT_VIDEO, PREVIEW_EVERY
from models import load_all_models
from court_detector import draw_court
from court_tracker import CourtTracker
from player_tracker import PlayerTracker, draw_players
from ball_tracker import BallTracker
from mini_court import draw_mini_court
from bounce_detector import BounceDetector
from scene_cut import SceneMonitor
from video_io import ThreadedVideoReader, ThreadedVideoWriter, open_video_io


def _reset_scene(court_tracker, ball_tracker, person_tracker, bounce_detector, reset_court=True):
    """Cảnh mới: bỏ toàn bộ trạng thái của cảnh cũ (reset_court=False khi sân vừa được detect lại)."""
    if reset_court:
        court_tracker.reset()

    ball_tracker.reset()
    person_tracker.reset()

    if bounce_detector is not None:
        bounce_detector.reset()


def process_frame(frame, court_tracker, ball_tracker, person_tracker, scene_cut=None, bounce_detector=None,
                  aux=None):
    """Chạy toàn bộ pipeline (court, players, ball, mini-map) cho 1 frame.
    aux: dict tiền xử lý tính sẵn ở thread đọc (xem video_io.make_aux), None thì tự tính."""

    aux = aux or {}
    annotated_frame = frame.copy()

    # 0. Chuyển cảnh theo hình ảnh (cut / dissolve): bỏ sân / vùng tìm bóng / player cũ
    if scene_cut is not None and scene_cut.update(frame, aux.get("small")):
        _reset_scene(court_tracker, ball_tracker, person_tracker, bounce_detector)

    # 1. Court
    court, H = court_tracker.update(frame, aux.get("gray"))

    # 2. Players (đúng 1 người / nửa sân)
    players = []

    if court is not None:
        draw_court(annotated_frame, court)
        players = person_tracker.update(frame, H, court[0])
        draw_players(annotated_frame, players)

    # 3. Ball
    ball_point = ball_tracker.detect(frame, annotated_frame)

    # 3b. Chuyển cảnh theo tín hiệu: mất sân + mất bóng + không có player một lúc, hoặc sân xuất hiện lại
    event = (scene_cut.update_signals(court is not None, ball_point is not None, len(players) > 0)
             if scene_cut is not None else None)

    if event is not None:
        _reset_scene(court_tracker, ball_tracker, person_tracker, bounce_detector,
                     reset_court=(event == "lost"))
        ball_point = None

    # 4. Bóng nảy
    swing_halves = set()

    if bounce_detector is not None:
        bounce_detector.update(ball_point, players, H if court is not None else None)
        swing_halves = bounce_detector.recent_hit_halves()   # cổ tay HOẶC quỹ đạo bóng

    # 5. Mini-map (ký hiệu HIT quanh player đang vung vợt)
    if court is not None:
        draw_mini_court(
            annotated_frame, H, players, ball_point,
            marks=bounce_detector.recent_marks() if bounce_detector is not None else None,
            swing_halves=swing_halves,
        )

    # 6. Báo chuyển cảnh
    if scene_cut is not None:
        scene_cut.draw(annotated_frame)

    return annotated_frame


def run(input_video=INPUT_VIDEO, output_video=OUTPUT_VIDEO, show_preview=True,
        models=None, max_frames=None):
    """
    Args:
        models:     tuple do load_all_models() trả về (truyền vào để tái sử dụng, vd khi benchmark)
        max_frames: dừng sau N frame (None = chạy hết video)

    Returns:
        list thời gian (giây) giữa 2 frame được xử lý liên tiếp.
    """

    if models is None:
        models = load_all_models()

    ball_model, person_model, court_model, ball_local_model = models

    court_tracker = CourtTracker(court_model)
    ball_tracker = BallTracker(ball_model, local_model=ball_local_model)
    person_tracker = PlayerTracker(person_model)

    cap, out = open_video_io(input_video, output_video)
    scene_cut = SceneMonitor(cap.get(cv2.CAP_PROP_FPS))

    bounce_detector = BounceDetector(cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    reader = ThreadedVideoReader(cap)
    writer = ThreadedVideoWriter(out)

    frame_times = []
    prev_time = time.time()

    try:
        while True:

            frame, aux = reader.read_aux()

            if frame is None:
                break

            annotated_frame = process_frame(
                frame, court_tracker, ball_tracker, person_tracker, scene_cut, bounce_detector, aux)

            # FPS overlay
            curr_time = time.time()
            dt = max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            frame_times.append(dt)

            cv2.putText(
                annotated_frame, f"FPS: {1.0 / dt:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
            )

            writer.write(annotated_frame)

            # Preview tốn vài ms/frame (resize 1080p + imshow): chỉ hiện mỗi PREVIEW_EVERY frame
            if show_preview and len(frame_times) % PREVIEW_EVERY == 0:
                cv2.imshow("Tennis Analysis", cv2.resize(annotated_frame, None, fx=0.7, fy=0.7))

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and len(frame_times) >= max_frames:
                break

    finally:
        reader.stop()
        writer.close()
        cap.release()
        court_tracker.close()
        cv2.destroyAllWindows()

    print("Saved:", output_video)

    return frame_times


def parse_args():
    p = argparse.ArgumentParser(description="Tennis Analysis: court / player / ball / bounce / swing trên video broadcast")
    p.add_argument("--input", default=INPUT_VIDEO, help="video đầu vào")
    p.add_argument("--output", default=None, help="video đầu ra (mặc định: <input>_result.mp4)")
    p.add_argument("--no-preview", action="store_true", help="không mở cửa sổ preview")
    p.add_argument("--max-frames", type=int, default=None, help="dừng sau N frame")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or os.path.splitext(args.input)[0] + "_result.mp4"
    run(args.input, output, show_preview=not args.no_preview, max_frames=args.max_frames)