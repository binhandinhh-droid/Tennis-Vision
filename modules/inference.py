import time
import cv2
import config
from models import load_all_models
from court_detector import draw_court
from court_tracker import CourtTracker
from player_tracker import track_players
from ball_tracker import BallTracker
from mini_court import draw_mini_court


def open_video_io(input_video: str, output_video: str):

    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    return cap, out, width, height


def process_frame(frame, court_tracker, ball_tracker, person_model):
    """Chạy toàn bộ pipeline (court, players, ball, mini-map) cho 1 frame.

    Returns:
        annotated_frame
    """

    annotated_frame = frame.copy()

    # 1. Court (cache + optical flow / detect định kỳ) -----------------
    cached_court, cached_H = court_tracker.update(frame)

    if cached_court is not None:
        draw_court(annotated_frame, cached_court[0], cached_court[1], cached_court[2])

    # 2. Players (chỉ giữ đúng 1 người / nửa sân) -----------------------
    mini_map_players = []

    if cached_court is not None:
        mini_map_players = track_players(
            frame, annotated_frame, person_model, cached_H,
            config.MINI_COURT_W, config.MINI_COURT_H,
        )

    # 3. Ball -------------------------------------------------------------
    mini_map_ball_point = ball_tracker.detect(frame, annotated_frame)

    # 4. Mini-map ----------------------------------------------------------
    if cached_court is not None:
        draw_mini_court(annotated_frame, cached_H, mini_map_players, mini_map_ball_point)

    return annotated_frame


def run(input_video=config.INPUT_VIDEO, output_video=config.OUTPUT_VIDEO, show_preview=True):

    ball_model, person_model, court_model = load_all_models()

    court_tracker = CourtTracker(court_model, config.MINI_COURT_W, config.MINI_COURT_H)
    ball_tracker = BallTracker(ball_model)

    cap, out, width, height = open_video_io(input_video, output_video)

    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame = process_frame(frame, court_tracker, ball_tracker, person_model)

        # --- FPS overlay ---
        curr_time = time.time()
        fps_now = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps_now:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        out.write(annotated_frame)

        if show_preview:
            display = cv2.resize(annotated_frame, None, fx=0.7, fy=0.7)
            cv2.imshow("Tennis Analysis", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved:", output_video)


if __name__ == "__main__":
    run()
