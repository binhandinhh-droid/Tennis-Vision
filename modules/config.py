"""All configuration / constants for the Tennis Analysis pipeline."""

import os
from pathlib import Path

import numpy as np
import torch

# Project root: all default paths are resolved from here (can be overridden with environment variables).
ROOT = Path(__file__).resolve().parent


def _path(env_name, default):
    return os.environ.get(env_name, str(default))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable to draw extra boxes (court box + score, ball search ROI, ball box, player ids).
DEBUG_MODE = False

# ---- Model paths ----
BALL_MODEL_PATH = _path("TENNIS_BALL_MODEL", ROOT / "weights" / "best_yolov26m.pt")
# Pose model (person box + wrist keypoints). A bare name makes ultralytics download it on first use.
# YOLO26x-pose is much more accurate than yolov8m-pose; if the download fails, PERSON_MODEL_FALLBACKS are tried.
PERSON_MODEL_PATH = _path("TENNIS_PERSON_MODEL", ROOT / "weights" / "yolo26x-pose.pt")
PERSON_MODEL_FALLBACKS = ["yolo11x-pose.pt", "yolo11l-pose.pt"]
PERSON_IMGSZ = 1280   # (.pt models only) imgsz for the crop around the court (far player is small -> needs to be large). TensorRT engine: export with exactly this imgsz
# ---- TensorRT engine shape (h, w) - RECTANGULAR so no compute is spent on padding ----
# The scale is the same as a square 1280 engine, so quality is unchanged; the GPU just skips the padding.
# Ball global: a 1080p frame is letterboxed to width 1280 -> height 720 -> rounded up to 736. Adjust for non-16:9 video.
BALL_ENGINE_IMGSZ = (736, 1280)
# Person: run `python measure_person_crop.py` to measure and fill in the recommended value (default (1280, 1280) = as before).
PERSON_ENGINE_IMGSZ = (736, 1280)

PREVIEW_EVERY = 3   # show the preview only every N frames (imshow + resize cost a few ms/frame)

COURT_MODEL_PATH = _path("TENNIS_COURT_MODEL", ROOT / "weights" / "best_tennis_court_checkpoint.pth")

NUM_KEYPOINTS = 14

# ---- Video I/O ----
INPUT_VIDEO = _path("TENNIS_INPUT_VIDEO", ROOT / "input_video" / "input_video.mp4")
OUTPUT_VIDEO = os.path.splitext(INPUT_VIDEO)[0] + "_result.mp4"   # same folder, "_result" suffix
IO_QUEUE_SIZE = 16   # max number of frames waiting in the read / write queue


# ---- Ball tracking: anti-jump configuration ----
BALL_SEARCH_MARGIN = 100

# Confidence thresholds
BALL_LOCAL_CONF = 0.25
BALL_GLOBAL_CONF = 0.5

# Tracking state
BALL_LOST_PATIENCE = 5
BALL_SEARCH_GROWTH = 0.35

# ---- Court cache / optical flow ----
# Re-detect the court (KeypointRCNN, background thread) ON DEMAND instead of on a fixed period:
COURT_MIN_INTERVAL = 15          # minimum frames between two detections (avoids constant detection while panning / zooming)
COURT_MAX_INTERVAL = 300         # safety upper bound: at most N frames without detecting, even if the camera is still
COURT_REDETECT_TRAVEL_PX = 40    # total distance keypoints drifted with the camera since the last detection
COURT_FLOW_MIN_INLIER_RATIO = 0.8   # fraction of points consistent with the camera transform; lower -> re-detect
COURT_FLOW_MAX_RESIDUAL_PX = 3.0    # fit error (px) above this -> camera is zooming / changing angle, re-detect
COURT_FLOW_DEADBAND_PX = 0.3        # shift below this counts as a still camera: keep the court as is, do not accumulate noise
OPTICAL_FLOW_SCALE = 0.5         # optical flow runs on a grayscale image downscaled by this ratio
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(3, 30, 0.01))

# Fraction of successfully tracked points below this -> track is lost, re-detect immediately
COURT_TRACK_MIN_VALID_RATIO = 0.6
COURT_DETECT_SCORE_THRESH = 0.5
COURT_ACQUIRE_SCORE = 0.8          # synchronous detection (just after a scene change / lost track): higher threshold to avoid accepting a non-court scene

# Check that the keypoint set really is a court (rejects phantom detections on faces, stands, ads...)
COURT_MIN_AREA_RATIO = 0.03        # minimum area of the 4-corner court quadrilateral relative to the frame
COURT_MAX_AREA_RATIO = 4.0         # maximum (zoomed-in court with corners outside the frame)
COURT_INSIDE_TOL = 0.04            # inner keypoints must lie inside the 4-corner quadrilateral (tolerance = N * sqrt(area))
COURT_INSIDE_MIN_RATIO = 0.8       # minimum fraction of inner keypoints that must lie inside the quadrilateral
COURT_EDGE_CONTRAST = 25           # a white line must be brighter than its surroundings by at least N gray levels
COURT_EDGE_MIN_FRAC = 0.5          # minimum fraction of sample points on a court edge that look like a white line
COURT_EDGE_MIN_GOOD = 2            # at least N/4 edges must look like white lines (if enough edges can be checked)
COURT_LOST_RETRY_INTERVAL = 5   # no court yet (just after a scene change...): retry detection only every N frames

# ---- Keypoint refinement with classical CV ----
REFINE_CROP_HALF = 27
REFINE_WHITE_THRESH = 170
REFINE_MAX_SHIFT = 25

# ---- Valid player movement area (meters) ----
PLAYER_MARGIN_SIDE_M = 1
PLAYER_MARGIN_DEPTH_M = 5.0
COURT_WIDTH_M = 10.97   # doubles court
COURT_LENGTH_M = 23.77

# Only feed the area around the court to the person YOLO (margin as a fraction of the court box size)
PERSON_CROP_MARGIN_X = 0.15
PERSON_CROP_MARGIN_Y = 0.25

# ---- Keeping players stable across frames (PlayerMemory) ----
PERSON_CONF = 0.2                  # person YOLO confidence threshold (default 0.25). Lower to catch the far player; the memory handles the noise
PLAYER_COAST_FRAMES = 12           # if YOLO misses a player, keep the old position for at most N frames before dropping it
PLAYER_MATCH_DIST_H = 0.8          # SAME person if box centers differ by < N * box height (even when the tracker changes id)
PLAYER_MATCH_GROWTH = 0.25         # the longer the miss, the wider the match region (each missed frame adds this ratio)
PLAYER_SWITCH_FRAMES = 8           # another person must be clearly better for N consecutive frames before replacing the tracked one
PLAYER_SWITCH_PRIORITY_GAP = 10.0  # "clearly better": at least N mini-map px closer to the court

PLAYER_COLORS = {"Player 1": (0, 255, 0), "Player 2": (255, 128, 0)}

# ---- Mini-map ----
MINI_COURT_W = 170
MINI_COURT_H = 380
MINI_PADDING = 14
MINI_MARGIN_TOP = 20
MINI_MARGIN_RIGHT = 20

SINGLES_INSET_RATIO = 1.37 / 10.97
SERVICE_LINE_RATIO_NEAR = 5.485 / 23.77
SERVICE_LINE_RATIO_FAR = 1 - SERVICE_LINE_RATIO_NEAR

# Valid player margin converted to mini-map units
_PLAYER_MARGIN_X = MINI_COURT_W * (PLAYER_MARGIN_SIDE_M / COURT_WIDTH_M)
_PLAYER_MARGIN_Y = MINI_COURT_H * (PLAYER_MARGIN_DEPTH_M / COURT_LENGTH_M)

# 4 normalized court corners on the mini-map (TL, TR, BR, BL)
_MINI_DST = np.array(
    [[0, 0], [MINI_COURT_W, 0], [MINI_COURT_W, MINI_COURT_H], [0, MINI_COURT_H]],
    dtype=np.float32,
)


# imgsz the ball model was trained at (best_1280.pt). Used to compute the ROI imgsz so the ball has the same
# pixel size as during training. Check with: YOLO(BALL_MODEL_PATH).overrides.get("imgsz")
BALL_TRAIN_IMGSZ = 1280
BALL_LOCAL_IMGSZ_MIN = 192     # smallest imgsz for the ROI (multiple of 32)
BALL_LOCAL_IMGSZ_MAX = 640     # largest imgsz for the ROI
BALL_LOCAL_MAX_MARGIN = 160    # cap on the ROI margin when the ball is lost (the margin grows but never exceeds this);
                               # the larger the ROI, the more the ball shrinks when resized to a small imgsz


# ---- Bounce detection. Pixel units are at width 1920 (auto-scaled to the video resolution). ----
# Starting values: enable DEBUG_MODE to see why each candidate was rejected, then tune for your video.
BALL_EVENT_MAX_GAP_S = 0.40        # if two ball sightings are further apart than this, drop the old trajectory
BOUNCE_HALF_WINDOW_S = 0.12        # half window of the trajectory around the bounce point (seconds); larger -> more noise tolerant but more delay
BOUNCE_MIN_DROP_PX = 1.5           # the ball must go down / up by at least N px (image) on both sides of the bottom. Lower if far or weak bounces are missed
BALL_BOUNCE_COOLDOWN_S = 0.10      # minimum time between two consecutive bounces
BOUNCE_HIT_SPEED_GAIN = 1.5        # within racket reach: speed after / before above this (or horizontal reversal) = a hit
HIT_REACH_X = 0.75                 # racket reach, as a fraction of player box height: horizontal
HIT_REACH_UP = 0.30                #   above the head
BALL_AIR_MIN_RAISE = 0.15           # ball higher than the feet by >15% of player height, within reach -> considered a hit
BOUNCE_MIN_DVY_PX = 2.0            # vy must jump upward by at least N px/frame (at 1920) to count as an impulse-type bounce (oblique bounce). Too many false bounces -> raise to 3.0; missing some -> lower to 1.5
BOUNCE_MAX_OUTSIDE_PX = 100        # a bounce landing farther than this from the court (mini-map px) is treated as noise and ignored
BOUNCE_MARK_S = 3.0                # how long the X mark stays on the mini-map

# ---- Rejecting "fake bounces" caused by hits (hit log) ----
HIT_SIDE_EXCL_S = 1.0       # after a hit by A, for N seconds do not accept a bounce on A's half (the hit goes to the opponent's side)
HIT_AFTER_TOL_S = 0.10      # a hit registered up to N seconds later than the bounce candidate frame still counts as related
HIT_EVENT_KEEP_S = 4.0      # how long to keep the hit log
# Hit detection from the ball TRAJECTORY (no skeleton needed): ball within racket reach, approaches then leaves
# the player, direction reverses and does NOT lose speed the way a bounce does.
HIT_TRAJ_ENABLE = True
HIT_TRAJ_MIN_SPEED_PX = 2.0      # ball speed (px/frame at 1920) before AND after must be >= this (rejects a stationary ball / jitter)
HIT_TRAJ_TURN_DEG = 100.0        # angle between incoming and outgoing velocity >= N degrees counts as "reversed". Missing hits -> lower to 80; false hits -> raise to 120
HIT_TRAJ_SPEED_RATIO = 0.8       # speed after / before >= this (a bounce loses speed, so it is usually below this)
HIT_TRAJ_GROUND_RATIO = 1.2      # ball near the feet (easily confused with a bounce): require speed after / before >= this
HIT_TRAJ_APPROACH_H = 0.08       # the ball must approach then leave the player center by at least N * person height at both ends of the window
HIT_DUP_S = 0.25                 # two hit signals from the same person closer than N seconds count as the same hit

# ---- Swing detection (wrist keypoints from YOLO-pose) ----
KP_WRISTS = (9, 10)                # COCO: left_wrist, right_wrist
KP_TORSO = (5, 6, 11, 12)          # COCO: left/right shoulder, left/right hip. The torso center is the reference for the wrists (removes camera motion / running)
KP_TORSO_MIN = 2                   # need at least N confident torso keypoints to compute the torso center
KP_MIN_CONF = 0.4                  # minimum keypoint confidence (wrist, torso)
SWING_SPEED_H_PER_S = 3.0          # wrist speed RELATIVE TO THE TORSO (person heights / second) that counts as a swing. Racket swing ~4-8, arm swing while running ~1.5. Missing hits -> lower to 2.5; false hits -> raise to 4.0
SWING_MAX_SPEED_H_PER_S = 15.0     # faster than this is a keypoint jump / left-right flip, not real motion: discarded
SWING_MIN_FRAMES = 2               # the wrist must be fast for at least N consecutive frames to count as a swing (rejects 1-frame noise)
SWING_BALL_DIST_H = 0.6            # ball closer than N * person height to the wrist means that swing "hits" the ball. Serves / smashes may need a larger value
SWING_TOL_S = 0.10                 # allowed time difference between a swing and the bounce candidate frame (seconds)
SWING_SPEED_SPAN = 2               # wrist speed is measured over N frames (reduces keypoint jitter)
SWING_SLOW_RATIO = 0.6             # the wrist must be slower than ratio * SWING_SPEED_H_PER_S (and near the ball) before we dare to assert NO swing
SWING_SHOW_S = 0.4                 # how long to keep the "SWING!" signal on screen after a swing (seconds)
SWING_RELEVANT_DIST_H = 1.0       # only consider wrists closer than N * person height to the ball when concluding "no swing"


# Scene change (reset court / ball tracker). Detected from histogram / image difference between two consecutive frames,
# and (for dissolve / wipe - gradual transitions) from the histogram difference versus N frames ago.
SCENE_CUT_HIST_CORR = 0.7          # HSV histogram correlation below this -> scene change
SCENE_CUT_MAD = 45.0               # or mean grayscale image difference (0-255) above this
SCENE_CUT_WINDOW = 10              # compare with the frame N frames back to catch gradual transitions
SCENE_DISSOLVE_CORR = 0.6          # correlation versus N frames ago below this -> gradual scene change

# Scene change from SIGNALS (besides histogram comparison): non-court scenes (replay, close-up, ads...)
SCENE_LOST_S = 0.5            # court lost AND ball not seen for N consecutive seconds -> assume we switched to a scene without a court
SCENE_LOST_BALL_RATIO = 0.2   # within those N seconds, the fraction of frames with a ball must be <= this (tolerates 1-2 false ball detections)
SCENE_BANNER_S = 1.5          # how long the "SCENE CHANGE" text stays on screen
COURT_DROP_AFTER_FAILS = 2    # synchronous court detection fails (confidence < COURT_DETECT_SCORE_THRESH) N times in a row -> drop the old court

# ---- Quantization (falls back to FP32 automatically if the file is missing) ----
USE_TENSORRT_BALL = False    # enable after exporting the engine (see export_engines.py)
USE_TENSORRT_PERSON = False  # enable after exporting the engine (see export_engines.py)
USE_FP16_COURT = False

BALL_MODEL_ENGINE_PATH = BALL_MODEL_PATH.replace(".pt", ".engine")
PERSON_MODEL_ENGINE_PATH = PERSON_MODEL_PATH.replace(".pt", ".engine")
COURT_MODEL_FP16_PATH = COURT_MODEL_PATH.replace(".pth", "_fp16.pth")

# Small engine (low imgsz, FP16) used only for the local search around the ball (export_engines.py, EXPORT_BALL_LOCAL = True).
USE_BALL_LOCAL_ENGINE = False
BALL_LOCAL_ENGINE_PATH = BALL_MODEL_PATH.replace(".pt", "_local.engine")
BALL_LOCAL_ENGINE_IMGSZ = 192   # imgsz used when exporting the ROI engine (export_engines.py)
