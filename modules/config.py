import os
import torch

# =====================================================
# Device
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bật DEBUG_MODE lên để xem thêm mấy cái box phụ (điểm số sân, vùng ROI tìm bóng...)
# Riêng số thứ tự keypoint thì lúc nào cũng vẽ, không liên quan tới cờ này.
DEBUG_MODE = False

# =====================================================
# Model paths
# =====================================================

BALL_MODEL_PATH = r"D:\project\tennis_analysis\weights\best_1280.pt"
PERSON_MODEL_PATH = r"D:\project\tennis_analysis\weights\yolov8m.pt"
COURT_MODEL_PATH = r"D:\project\tennis_analysis\weights\best_tennis_court_checkpoint.pth"

NUM_KEYPOINTS = 14

# =====================================================
# Video I/O
# =====================================================
INPUT_VIDEO = r"D:\project\tennis_analysis\input_video\input_video_3.mp4"

def build_output_path(input_video: str) -> str:
    # Video output để cùng thư mục với input, chỉ thêm hậu tố "_result"
    folder = os.path.dirname(input_video)
    name = os.path.splitext(os.path.basename(input_video))[0]
    return os.path.join(folder, f"{name}_result.mp4")


OUTPUT_VIDEO = build_output_path(INPUT_VIDEO)

# =====================================================
# Các cặp keypoint để nối thành đường kẻ sân (khi cần vẽ khung dây)
# =====================================================

COURT_LINES = [
    # Khung ngoài
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),

    # Đường biên đơn 2 bên
    (4, 5),
    (6, 7),

    # Đường giao bóng
    (8, 9),
    (10, 11),

    # Đường giữa sân
    (12, 13),
]

# =====================================================
# Ball tracking
# =====================================================

BALL_SEARCH_MARGIN = 120   # bán kính vùng tìm bóng quanh vị trí frame trước
BALL_LOCAL_CONF = 0.30     # ngưỡng conf khi tìm trong vùng nhỏ (local search)
BALL_GLOBAL_CONF = 0.40    # ngưỡng conf khi phải quét lại toàn frame

# =====================================================
# Cache sân / tracking bằng optical flow
# =====================================================

# Cứ mỗi bao nhiêu frame thì chạy lại model KeypointRCNN để detect sân 1 lần
COURT_UPDATE_INTERVAL = 30

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(3, 30, 0.01),
)

# Nếu tỉ lệ điểm track thành công (status==1) tụt xuống dưới ngưỡng này
# thì coi như mất track luôn (thường do cắt cảnh / camera xoay đột ngột),
# lúc đó phải detect lại ngay chứ không đợi đến chu kỳ COURT_UPDATE_INTERVAL kế tiếp
COURT_TRACK_MIN_VALID_RATIO = 0.6

COURT_DETECT_SCORE_THRESH = 0.5

# =====================================================
# Refine bằng CV cổ điển (kéo keypoint dính sát vào giao điểm đường kẻ)
# =====================================================

REFINE_CROP_HALF = 20      # crop 1 ô vuông quanh keypoint để dò đường trắng
REFINE_WHITE_THRESH = 170  # ngưỡng độ sáng để coi là "vạch trắng"
REFINE_MAX_SHIFT = 25      # dịch tối đa bao nhiêu px, tránh nhảy lung tung

# =====================================================
# Giới hạn vùng di chuyển của player (tính theo mét thật, quy đổi qua homography)
# =====================================================

PLAYER_MARGIN_SIDE_M = 1      # nới thêm 2 bên theo chiều rộng sân
PLAYER_MARGIN_DEPTH_M = 5.0   # nới thêm phía sau baseline

COURT_WIDTH_M = 10.97   # chiều rộng sân đôi
COURT_LENGTH_M = 23.77  # chiều dài sân

# =====================================================
# Mini-map sân (top-down view) đặt ở góc trên bên phải màn hình
# =====================================================

MINI_COURT_W = 170      # bề rộng sân trên mini-map (px)
MINI_COURT_H = 380      # chiều dài sân trên mini-map (px)
MINI_PADDING = 14       # đệm quanh viền sân trong khung mini-map
MINI_MARGIN_TOP = 20    # cách mép trên của frame bao nhiêu px
MINI_MARGIN_RIGHT = 20  # cách mép phải của frame bao nhiêu px

# Tỉ lệ theo kích thước sân chuẩn ITF, dùng để vẽ đúng vị trí
# đường biên đơn, đường giao bóng và đường giữa sân trên mini-map
SINGLES_INSET_RATIO = 1.37 / 10.97          # lề đơn tính từ biên đôi vào
SERVICE_LINE_RATIO_NEAR = 5.485 / 23.77     # khoảng cách tới vạch giao bóng gần
SERVICE_LINE_RATIO_FAR = 1 - SERVICE_LINE_RATIO_NEAR

# =====================================================
# Lượng tử hóa model (TensorRT INT8 cho YOLO, FP16 cho KeypointRCNN)
# =====================================================

# Bật để dùng bản đã lượng tử hóa cho nhanh. Nếu chưa chạy script quantize
# tương ứng nên chưa có file .engine / _fp16.pth, models.py sẽ tự fallback
# về bản gốc FP32 và in cảnh báo ra console, không bị crash đâu.
USE_TENSORRT_BALL = True
USE_TENSORRT_PERSON = True
USE_FP16_COURT = True

BALL_MODEL_ENGINE_PATH = BALL_MODEL_PATH.replace(".pt", ".engine")
PERSON_MODEL_ENGINE_PATH = PERSON_MODEL_PATH.replace(".pt", ".engine")
COURT_MODEL_FP16_PATH = COURT_MODEL_PATH.replace(".pth", "_fp16.pth")

# Thư mục chứa frame calibration trích ra từ INPUT_VIDEO, dùng cho bước
# quantize INT8 của 2 model YOLO (xem extract_calibration_frames.py)
CALIBRATION_DATA_DIR = r"D:\project\tennis_analysis\calibration_data"
CALIBRATION_NUM_FRAMES = 300