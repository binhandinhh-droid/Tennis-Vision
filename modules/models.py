import os

import torch
import torchvision
from ultralytics import YOLO
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

import config


def build_court_model(num_keypoints: int) -> torch.nn.Module:
    """Tạo kiến trúc KeypointRCNN với đầu ra num_keypoints (chưa load weight)."""

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None)

    in_features = (
        model.roi_heads
        .keypoint_predictor
        .kps_score_lowres
        .in_channels
    )

    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_features,
        num_keypoints,
    )

    return model


def load_court_model(weights_path: str, num_keypoints: int, device: torch.device,
                      use_fp16: bool = False) -> torch.nn.Module:
    """
    Load court model. Nếu use_fp16=True, weights_path phải trỏ tới file
    đã được quantize_court_model.py tạo ra (state_dict FP16) -> model
    sẽ được chạy ở half precision. Ngược lại load bản FP32 bình thường.
    """

    model = build_court_model(num_keypoints)

    # Mở file checkpoint
    checkpoint = torch.load(weights_path, map_location=device)

    # Kiểm tra nếu checkpoint là một dict chứa 'model_state_dict' thì lấy ra, ngược lại dùng trực tiếp
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load state_dict vào mô hình
    model.load_state_dict(state_dict)

    model.to(device)

    if use_fp16:
        model.half()

    model.eval()

    return model


def _resolve_yolo_weights(pt_path: str, engine_path: str, use_engine: bool, label: str) -> str:
    """Chọn file .engine (TensorRT INT8) nếu use_engine=True và file đó
    thực sự tồn tại trên đĩa, ngược lại fallback về .pt gốc + cảnh báo."""

    if use_engine and os.path.exists(engine_path):
        print(f"[{label}] Dùng bản đã lượng tử hóa (TensorRT INT8): {engine_path}")
        return engine_path

    if use_engine and not os.path.exists(engine_path):
        print(
            f"[{label}] CẢNH BÁO: chưa tìm thấy '{engine_path}'. "
            f"Chạy quantize_yolo_models.py trước để tạo engine. "
            f"Tạm thời dùng bản gốc FP32: {pt_path}"
        )

    return pt_path


def load_all_models():
    """
    Load toàn bộ model cần thiết cho pipeline, ưu tiên bản đã lượng tử
    hóa nếu tồn tại (bật/tắt qua config.USE_TENSORRT_*, config.USE_FP16_COURT).
    Tự động fallback về bản FP32 gốc nếu chưa chạy script quantize
    tương ứng, để không bao giờ bị crash vì thiếu file.

    Returns:
        ball_model, person_model, court_model
    """

    ball_weights = _resolve_yolo_weights(
        config.BALL_MODEL_PATH, config.BALL_MODEL_ENGINE_PATH,
        config.USE_TENSORRT_BALL, "Ball model",
    )
    person_weights = _resolve_yolo_weights(
        config.PERSON_MODEL_PATH, config.PERSON_MODEL_ENGINE_PATH,
        config.USE_TENSORRT_PERSON, "Person model",
    )

    ball_model = YOLO(ball_weights)
    person_model = YOLO(person_weights)

    use_fp16_court = config.USE_FP16_COURT and os.path.exists(config.COURT_MODEL_FP16_PATH)

    if config.USE_FP16_COURT and not use_fp16_court:
        print(
            f"[Court model] CẢNH BÁO: chưa tìm thấy '{config.COURT_MODEL_FP16_PATH}'. "
            f"Chạy quantize_court_model.py trước. Tạm thời dùng bản gốc FP32."
        )

    court_weights_path = config.COURT_MODEL_FP16_PATH if use_fp16_court else config.COURT_MODEL_PATH

    court_model = load_court_model(
        court_weights_path,
        config.NUM_KEYPOINTS,
        config.DEVICE,
        use_fp16=use_fp16_court,
    )

    print(f"Court model loaded ({'FP16' if use_fp16_court else 'FP32'}).")

    return ball_model, person_model, court_model
