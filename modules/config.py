"""Toàn bộ cấu hình / hằng số của pipeline Tennis Analysis."""

import os
from pathlib import Path

import numpy as np
import torch

# Thư mục gốc của project: mọi đường dẫn mặc định tính từ đây (có thể ghi đè bằng biến môi trường).
ROOT = Path(__file__).resolve().parent


def _path(env_name, default):
    return os.environ.get(env_name, str(default))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bật để xem thêm box phụ (box sân + điểm số, vùng ROI tìm bóng, box bóng, id player).
DEBUG_MODE = False

# ---- Model paths ----
BALL_MODEL_PATH = _path("TENNIS_BALL_MODEL", ROOT / "weights" / "best_yolov26m.pt")
# Model pose (box người + keypoint cổ tay). Tên trần -> ultralytics tự tải về lần đầu.
# YOLO26x-pose chính xác hơn hẳn yolov8m-pose; nếu tải không được sẽ thử PERSON_MODEL_FALLBACKS.
PERSON_MODEL_PATH = _path("TENNIS_PERSON_MODEL", ROOT / "weights" / "yolo26x-pose.pt")
PERSON_MODEL_FALLBACKS = ["yolo11x-pose.pt", "yolo11l-pose.pt"]
PERSON_IMGSZ = 1280   # (chỉ dùng cho model .pt) imgsz cho vùng crop quanh sân (player ở xa nhỏ -> cần lớn). Engine TensorRT: export đúng imgsz này
# ---- Shape engine TensorRT (h, w) - HÌNH CHỮ NHẬT để không tính toán trên phần padding ----
# Scale giữ nguyên như engine vuông 1280 nên chất lượng không đổi, GPU chỉ bỏ phần padding.
# Ball global: frame 1080p letterbox về rộng 1280 -> cao 720 -> làm tròn 736. Video khác tỉ lệ 16:9 thì sửa lại.
BALL_ENGINE_IMGSZ = (736, 1280)
# Person: chạy `python measure_person_crop.py` để đo rồi điền giá trị khuyến nghị (mặc định (1280, 1280) = như cũ).
PERSON_ENGINE_IMGSZ = (736, 1280)

PREVIEW_EVERY = 3   # chỉ hiện preview mỗi N frame (imshow + resize tốn vài ms/frame)

COURT_MODEL_PATH = _path("TENNIS_COURT_MODEL", ROOT / "weights" / "best_tennis_court_checkpoint.pth")

NUM_KEYPOINTS = 14

# ---- Video I/O ----
INPUT_VIDEO = _path("TENNIS_INPUT_VIDEO", ROOT / "input_video" / "input_video.mp4")
OUTPUT_VIDEO = os.path.splitext(INPUT_VIDEO)[0] + "_result.mp4"   # cùng thư mục, hậu tố "_result"
IO_QUEUE_SIZE = 16   # số frame tối đa nằm chờ trong queue đọc / ghi



# ---- Ball tracking: anti-jump configuration ----
BALL_SEARCH_MARGIN = 100

# Confidence threshold
BALL_LOCAL_CONF = 0.25
BALL_GLOBAL_CONF = 0.5

# Tracking state
BALL_LOST_PATIENCE = 5
BALL_SEARCH_GROWTH = 0.35

# ---- Court cache / optical flow ----
# Detect lại sân (KeypointRCNN, thread nền) theo NHU CẦU thay vì chu kỳ cố định:
COURT_MIN_INTERVAL = 15          # tối thiểu N frame giữa 2 lần detect (tránh detect liên tục khi pan/zoom)
COURT_MAX_INTERVAL = 300         # chặn trên an toàn: tối đa N frame không detect dù camera đứng yên
COURT_REDETECT_TRAVEL_PX = 40    # tổng quãng đường keypoint trôi theo camera kể từ lần detect cuối
COURT_FLOW_MIN_INLIER_RATIO = 0.8   # tỉ lệ điểm khớp với phép biến đổi camera, thấp hơn -> detect lại
COURT_FLOW_MAX_RESIDUAL_PX = 3.0    # sai số khớp (px) cao hơn -> camera đang zoom/đổi góc, detect lại
COURT_FLOW_DEADBAND_PX = 0.3        # dịch < ngưỡng này coi như camera đứng yên: giữ nguyên sân, không cộng nhiễu
OPTICAL_FLOW_SCALE = 0.5         # optical flow chạy trên ảnh xám thu nhỏ theo tỉ lệ này
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(3, 30, 0.01))

# Tỉ lệ điểm track thành công thấp hơn ngưỡng này -> coi như mất track, detect lại ngay
COURT_TRACK_MIN_VALID_RATIO = 0.6
COURT_DETECT_SCORE_THRESH = 0.5
COURT_ACQUIRE_SCORE = 0.8          # detect đồng bộ (vừa chuyển cảnh / mất track): ngưỡng cao hơn, tránh nhận nhầm cảnh không phải sân

# Kiểm tra bộ keypoint có thật sự là sân không (loại detect ảo trên mặt người, khán đài, quảng cáo...)
COURT_MIN_AREA_RATIO = 0.03        # diện tích tứ giác 4 góc sân tối thiểu so với frame
COURT_MAX_AREA_RATIO = 4.0         # tối đa (sân zoom to có góc ngoài khung hình)
COURT_INSIDE_TOL = 0.04            # keypoint bên trong phải nằm trong tứ giác 4 góc (sai số = N * căn diện tích)
COURT_INSIDE_MIN_RATIO = 0.8       # tỉ lệ keypoint bên trong nằm trong tứ giác tối thiểu
COURT_EDGE_CONTRAST = 25           # vạch trắng phải sáng hơn nền xung quanh ít nhất N mức xám
COURT_EDGE_MIN_FRAC = 0.5          # tỉ lệ điểm mẫu trên cạnh sân có dạng vạch trắng tối thiểu
COURT_EDGE_MIN_GOOD = 2            # ít nhất N/4 cạnh phải nhìn ra vạch trắng (nếu đủ cạnh để kiểm tra)
COURT_LOST_RETRY_INTERVAL = 5   # chưa có sân (vừa chuyển cảnh...): chỉ thử detect lại mỗi N frame

# ---- Refine keypoint bằng CV cổ điển ----
REFINE_CROP_HALF = 27
REFINE_WHITE_THRESH = 170
REFINE_MAX_SHIFT = 25

# ---- Vùng di chuyển hợp lệ của player (mét) ----
PLAYER_MARGIN_SIDE_M = 1
PLAYER_MARGIN_DEPTH_M = 5.0
COURT_WIDTH_M = 10.97   # sân đôi
COURT_LENGTH_M = 23.77

# Chỉ đưa vùng quanh sân vào YOLO người (margin tính theo tỉ lệ kích thước box sân)
PERSON_CROP_MARGIN_X = 0.15
PERSON_CROP_MARGIN_Y = 0.25

# ---- Giữ ổn định player qua các frame (PlayerMemory) ----
PERSON_CONF = 0.2                  # ngưỡng tin cậy YOLO người (mặc định 0.25). Thấp hơn để bắt player ở xa, bộ nhớ lo phần nhiễu
PLAYER_COAST_FRAMES = 12           # YOLO hụt player thì giữ nguyên vị trí cũ tối đa N frame rồi mới bỏ
PLAYER_MATCH_DIST_H = 0.8          # coi là CÙNG 1 người nếu tâm box lệch < N * chiều cao box (kể cả khi tracker đổi id)
PLAYER_MATCH_GROWTH = 0.25         # hụt càng lâu thì vùng khớp càng rộng (mỗi frame hụt cộng thêm tỉ lệ này)
PLAYER_SWITCH_FRAMES = 8           # người khác phải tốt hơn hẳn liên tục N frame thì mới thay người đang theo dõi
PLAYER_SWITCH_PRIORITY_GAP = 10.0  # "tốt hơn hẳn": gần sân hơn ít nhất N px mini-map

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

# Margin hợp lệ của player quy ra đơn vị mini-map
_PLAYER_MARGIN_X = MINI_COURT_W * (PLAYER_MARGIN_SIDE_M / COURT_WIDTH_M)
_PLAYER_MARGIN_Y = MINI_COURT_H * (PLAYER_MARGIN_DEPTH_M / COURT_LENGTH_M)

# 4 góc sân chuẩn hoá trên mini-map (TL, TR, BR, BL)
_MINI_DST = np.array(
    [[0, 0], [MINI_COURT_W, 0], [MINI_COURT_W, MINI_COURT_H], [0, MINI_COURT_H]],
    dtype=np.float32,
)


# imgsz lúc train model bóng (best_1280.pt). Dùng để tính imgsz cho ROI sao cho bóng có cùng
# kích thước pixel như lúc train. Kiểm tra bằng: YOLO(BALL_MODEL_PATH).overrides.get("imgsz")
BALL_TRAIN_IMGSZ = 1280
BALL_LOCAL_IMGSZ_MIN = 192     # imgsz nhỏ nhất cho ROI (bội số của 32)
BALL_LOCAL_IMGSZ_MAX = 640     # imgsz lớn nhất cho ROI
BALL_LOCAL_MAX_MARGIN = 160    # trần margin ROI khi mất bóng (margin nở dần nhưng không vượt mức này);
                               # ROI càng to thì bóng càng bị co khi resize về imgsz nhỏ




# ---- Phát hiện bóng nảy. Đơn vị px ở bề rộng 1920 (tự scale theo độ phân giải video). ----
# Giá trị khởi điểm: bật DEBUG_MODE để xem lý do mỗi lần bị bỏ rồi chỉnh theo video của bạn.
BALL_EVENT_MAX_GAP_S = 0.40        # 2 lần thấy bóng cách nhau quá lâu thì bỏ quỹ đạo cũ
BOUNCE_HALF_WINDOW_S = 0.12        # nửa cửa sổ quỹ đạo quanh điểm nảy (giây); tăng -> chịu nhiễu tốt hơn nhưng trễ hơn
BOUNCE_MIN_DROP_PX = 1.5           # bóng phải đi xuống / đi lên ít nhất N px (ảnh) ở 2 bên đáy. Giảm nếu sót nảy ở xa hoặc bóng nảy yếu
BALL_BOUNCE_COOLDOWN_S = 0.10      # 2 lần nảy liên tiếp cách nhau tối thiểu
BOUNCE_HIT_SPEED_GAIN = 1.5        # trong tầm vợt: tốc độ sau / trước lớn hơn mức này (hoặc đảo chiều ngang) = cú đánh
HIT_REACH_X = 0.75                 # tầm vợt, tính theo chiều cao box player: ngang
HIT_REACH_UP = 0.30                #   phía trên đầu
BALL_AIR_MIN_RAISE = 0.15           # bóng cao hơn chân >15% chiều cao người, trong tầm vợt -> xét là cú đánh
BOUNCE_MIN_DVY_PX = 2.0            # vy phải giật lên ít nhất N px/frame (ở 1920) để tính là nảy kiểu xung (bóng nảy xiên). Nảy giả nhiều -> tăng 3.0; còn sót -> giảm 1.5
BOUNCE_MAX_OUTSIDE_PX = 100        # nảy rơi xa sân hơn mức này (mini-map px) coi là nhiễu, bỏ qua
BOUNCE_MARK_S = 3.0                # thời gian hiện dấu X trên mini-map

# ---- Loại "nảy giả" do cú đánh (sổ cú đánh) ----
HIT_SIDE_EXCL_S = 1.0       # sau cú đánh của A, trong N giây không chấp nhận nảy ở nửa sân của A (cú đánh sang sân đối thủ)
HIT_AFTER_TOL_S = 0.10      # cú đánh được ghi nhận trễ tối đa N giây so với frame ứng viên nảy vẫn tính là liên quan
HIT_EVENT_KEEP_S = 4.0      # giữ sổ cú đánh bao lâu
# Phát hiện cú đánh từ QUỸ ĐẠO bóng (không cần skeleton): bóng trong tầm vợt, tiến sát rồi rời xa
# player, hướng đi đảo ngược và KHÔNG mất tốc như khi nảy.
HIT_TRAJ_ENABLE = True
HIT_TRAJ_MIN_SPEED_PX = 2.0      # tốc độ bóng (px/frame ở 1920) trước VÀ sau phải >= mức này (loại bóng đứng yên / jitter)
HIT_TRAJ_TURN_DEG = 100.0        # góc giữa vận tốc trước và sau >= N độ mới tính là "quay ngược lại". Sót cú đánh -> giảm 80; nhầm -> tăng 120
HIT_TRAJ_SPEED_RATIO = 0.8       # tốc độ sau / trước >= mức này (bóng nảy bị mất tốc nên thường < mức này)
HIT_TRAJ_GROUND_RATIO = 1.2      # bóng ở ngang chân (dễ lẫn với nảy): đòi tốc độ sau / trước >= mức này
HIT_TRAJ_APPROACH_H = 0.08       # bóng phải tiến sát rồi rời xa tâm player ít nhất N * chiều cao người ở 2 đầu cửa sổ
HIT_DUP_S = 0.25                 # 2 tín hiệu đánh của cùng 1 người cách nhau < N giây coi là cùng 1 cú

# ---- Phát hiện vung vợt (keypoint cổ tay từ YOLO-pose) ----
KP_WRISTS = (9, 10)                # COCO: left_wrist, right_wrist
KP_TORSO = (5, 6, 11, 12)          # COCO: vai trái/phải, hông trái/phải. Tâm thân là mốc để đo cổ tay (loại chuyển động camera / người chạy)
KP_TORSO_MIN = 2                   # cần ít nhất N keypoint thân tin cậy mới tính được tâm thân
KP_MIN_CONF = 0.4                  # độ tin cậy tối thiểu của keypoint (cổ tay, thân)
SWING_SPEED_H_PER_S = 3.0          # tốc độ cổ tay SO VỚI THÂN (chiều cao người / giây) tính là vung. Vung vợt ~4-8, đung đưa tay khi chạy ~1.5. Hụt cú đánh -> giảm 2.5; nhầm -> tăng 4.0
SWING_MAX_SPEED_H_PER_S = 15.0     # nhanh hơn mức này là keypoint nhảy / lệch trái-phải, không phải chuyển động thật: bỏ
SWING_MIN_FRAMES = 2               # cổ tay phải nhanh liên tiếp ít nhất N frame mới tính là vung (loại nhiễu 1 frame)
SWING_BALL_DIST_H = 0.6            # bóng cách cổ tay < N * chiều cao người thì cú vung đó "trúng" bóng. Giao bóng / smash có thể cần tăng
SWING_TOL_S = 0.10                 # sai lệch thời gian cho phép giữa swing và frame nảy ứng viên (giây)
SWING_SPEED_SPAN = 2               # tốc độ cổ tay đo trên N frame (giảm nhiễu keypoint rung)
SWING_SLOW_RATIO = 0.6             # cổ tay chậm hơn ratio * SWING_SPEED_H_PER_S (và ở gần bóng) mới dám khẳng định KHÔNG vung
SWING_SHOW_S = 0.4                 # giữ tín hiệu "SWING!" trên màn hình bao lâu sau cú vung (giây)
SWING_RELEVANT_DIST_H = 1.0       # chỉ xét cổ tay cách bóng < N * chiều cao người khi kết luận "không vung"



# Chuyển cảnh (reset court / ball tracker). Phát hiện bằng độ lệch histogram / ảnh giữa 2 frame liên tiếp,
# và (cho dissolve / wipe - chuyển cảnh mờ dần) bằng độ lệch histogram so với N frame trước.
SCENE_CUT_HIST_CORR = 0.7          # tương quan histogram HSV thấp hơn mức này -> chuyển cảnh
SCENE_CUT_MAD = 45.0               # hoặc chênh lệch ảnh xám trung bình (0-255) cao hơn mức này
SCENE_CUT_WINDOW = 10              # so với frame cách N frame để bắt chuyển cảnh mờ dần
SCENE_DISSOLVE_CORR = 0.6          # tương quan so với N frame trước thấp hơn mức này -> chuyển cảnh mờ dần

# Chuyển cảnh theo TÍN HIỆU (ngoài so sánh histogram): cảnh không phải sân (replay, cận mặt, quảng cáo...)
SCENE_LOST_S = 0.5            # mất sân VÀ không thấy bóng liên tục N giây -> coi là đã chuyển sang cảnh không có sân
SCENE_LOST_BALL_RATIO = 0.2   # trong N giây đó, tỉ lệ frame còn thấy bóng phải <= mức này (chịu được 1-2 detect bóng nhầm)
SCENE_BANNER_S = 1.5          # thời gian hiện chữ "SCENE CHANGE" trên màn hình
COURT_DROP_AFTER_FAILS = 2    # detect sân đồng bộ thất bại (độ tin cậy < COURT_DETECT_SCORE_THRESH) N lần liên tiếp -> bỏ sân cũ

# ---- Lượng tử hóa (tự fallback về FP32 nếu chưa có file) ----
USE_TENSORRT_BALL = False   # bật sau khi export engine (xem export_engines.py)
USE_TENSORRT_PERSON = False  # bật sau khi export engine (xem export_engines.py)
USE_FP16_COURT = False

BALL_MODEL_ENGINE_PATH = BALL_MODEL_PATH.replace(".pt", ".engine")
PERSON_MODEL_ENGINE_PATH = PERSON_MODEL_PATH.replace(".pt", ".engine")
COURT_MODEL_FP16_PATH = COURT_MODEL_PATH.replace(".pth", "_fp16.pth")

# Engine nhỏ (imgsz thấp, FP16) chỉ dùng cho local search quanh bóng (export_engines.py, EXPORT_BALL_LOCAL = True).
USE_BALL_LOCAL_ENGINE = False
BALL_LOCAL_ENGINE_PATH = BALL_MODEL_PATH.replace(".pt", "_local.engine")
BALL_LOCAL_ENGINE_IMGSZ = 192   # imgsz lúc export engine ROI (export_engines.py)