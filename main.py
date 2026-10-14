"""
Tennis Analysis - Inference

Pipeline: config -> models -> geometry -> court detector/tracker -> player tracker
-> ball tracker -> bounce detector -> mini-map -> main loop.

Scene changes (cut / dissolve) reset the court tracker, ball tracker and bounce detector.

Run:
    python main.py --input path/to/video.mp4

Keys in the preview window:  q to quit
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
    """New scene: drop all state of the old scene (reset_court=False when the court was just re-detected)."""
    if reset_court:
        court_tracker.reset()

    ball_tracker.reset()
    person_tracker.reset()

    if bounce_detector is not None:
        bounce_detector.reset()


def process_frame(frame, court_tracker, ball_tracker, person_tracker, scene_cut=None, bounce_detector=None,
                  aux=None):
    """Run the whole pipeline (court, players, ball, mini-map) on one frame.
    aux: dict of preprocessing precomputed in the reader thread (see video_io.make_aux), computed here if None."""

    aux = aux or {}
    annotated_frame = frame.copy()

    # 0. Visual scene change (cut / dissolve): drop the old court / ball search region / players
    if scene_cut is not None and scene_cut.update(frame, aux.get("small")):
        _reset_scene(court_tracker, ball_tracker, person_tracker, bounce_detector)

    # 1. Court
    court, H = court_tracker.update(frame, aux.get("gray"))

    # 2. Players (exactly 1 person per half)
    players = []

    if court is not None:
        draw_court(annotated_frame, court)
        players = person_tracker.update(frame, H, court[0])
        draw_players(annotated_frame, players)

    # 3. Ball
    ball_point = ball_tracker.detect(frame, annotated_frame)

    # 3b. Signal-based scene change: court + ball + players lost for a while, or the court reappears
    event = (scene_cut.update_signals(court is not None, ball_point is not None, len(players) > 0)
             if scene_cut is not None else None)

    if event is not None:
        _reset_scene(court_tracker, ball_tracker, person_tracker, bounce_detector,
                     reset_court=(event == "lost"))
        ball_point = None

    # 4. Ball bounce
    swing_halves = set()

    if bounce_detector is not None:
        bounce_detector.update(ball_point, players, H if court is not None else None)
        swing_halves = bounce_detector.recent_hit_halves()   # wrist OR ball trajectory

    # 5. Mini-map (HIT marker around the player who is swinging)
    if court is not None:
        draw_mini_court(
            annotated_frame, H, players, ball_point,
            marks=bounce_detector.recent_marks() if bounce_detector is not None else None,
            swing_halves=swing_halves,
        )

    # 6. Scene change banner
    if scene_cut is not None:
        scene_cut.draw(annotated_frame)

    return annotated_frame


def run(input_video=INPUT_VIDEO, output_video=OUTPUT_VIDEO, show_preview=True,
        models=None, max_frames=None):
    """
    Args:
        models:     tuple returned by load_all_models() (pass it in to reuse, e.g. when benchmarking)
        max_frames: stop after N frames (None = run the whole video)

    Returns:
        list of times (seconds) between two consecutively processed frames.
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

            # The preview costs a few ms/frame (1080p resize + imshow): only show every PREVIEW_EVERY frames
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
    p = argparse.ArgumentParser(
        description="Tennis Analysis: court / player / ball / bounce / swing detection on broadcast video")
    p.add_argument("--input", default=INPUT_VIDEO, help="input video")
    p.add_argument("--output", default=None, help="output video (default: <input>_result.mp4)")
    p.add_argument("--no-preview", action="store_true", help="do not open the preview window")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or os.path.splitext(args.input)[0] + "_result.mp4"
    run(args.input, output, show_preview=not args.no_preview, max_frames=args.max_frames)
