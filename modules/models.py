"""Load the models: ball (YOLO), person (YOLO), court (KeypointRCNN)."""

import os

import torch
import torchvision
from ultralytics import YOLO

from config import (
    DEVICE, NUM_KEYPOINTS,
    BALL_MODEL_PATH, BALL_MODEL_ENGINE_PATH, PERSON_MODEL_PATH, PERSON_MODEL_ENGINE_PATH,
    COURT_MODEL_PATH, COURT_MODEL_FP16_PATH,
    USE_TENSORRT_BALL, USE_TENSORRT_PERSON, USE_FP16_COURT,
    USE_BALL_LOCAL_ENGINE, BALL_LOCAL_ENGINE_PATH, PERSON_MODEL_FALLBACKS,
)


def load_court_model(weights_path, use_fp16=False):
    """KeypointRCNN with NUM_KEYPOINTS outputs + load weights."""

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=None, num_keypoints=NUM_KEYPOINTS
    )

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(DEVICE)

    if use_fp16:
        model.half()

    return model.eval()


def _resolve_yolo_weights(pt_path, engine_path, use_engine, label):
    """Pick the .engine (TensorRT) if enabled and the file exists, otherwise use the .pt."""

    if use_engine and os.path.exists(engine_path):
        print(f"[{label}] Using quantized model (TensorRT): {engine_path}")
        return engine_path

    if use_engine:
        print(f"[{label}] WARNING: '{engine_path}' not found. Falling back to the FP32 original: {pt_path}")

    return pt_path


def _load_person_model():
    """Try PERSON_MODEL_PATH, then each fallback in turn; use the first model that loads."""

    last_err = None

    for pt in [PERSON_MODEL_PATH, *PERSON_MODEL_FALLBACKS]:
        try:
            return YOLO(_resolve_yolo_weights(
                pt, pt.replace(".pt", ".engine"), USE_TENSORRT_PERSON, "Person model"))
        except Exception as e:  # noqa: BLE001
            print(f"[Person model] could not load '{pt}': {e}")
            last_err = e

    raise last_err


def load_all_models():
    """Returns: ball_model, person_model, court_model, ball_local_model (may be None)"""

    ball_model = YOLO(_resolve_yolo_weights(
        BALL_MODEL_PATH, BALL_MODEL_ENGINE_PATH, USE_TENSORRT_BALL, "Ball model"))
    person_model = _load_person_model()

    ball_local_model = None

    if USE_BALL_LOCAL_ENGINE:
        if os.path.exists(BALL_LOCAL_ENGINE_PATH):
            print(f"[Ball local] Using small engine for the ROI: {BALL_LOCAL_ENGINE_PATH}")
            ball_local_model = YOLO(BALL_LOCAL_ENGINE_PATH)
        else:
            print(
                f"[Ball local] WARNING: '{BALL_LOCAL_ENGINE_PATH}' not found. "
                f"Run export_engines.py (EXPORT_BALL_LOCAL = True) to create it. Falling back to the shared global model."
            )

    use_fp16_court = USE_FP16_COURT and os.path.exists(COURT_MODEL_FP16_PATH)

    if USE_FP16_COURT and not use_fp16_court:
        print(f"[Court model] WARNING: '{COURT_MODEL_FP16_PATH}' not found. Falling back to the FP32 original.")

    court_model = load_court_model(
        COURT_MODEL_FP16_PATH if use_fp16_court else COURT_MODEL_PATH,
        use_fp16=use_fp16_court,
    )
    print(f"Court model loaded ({'FP16' if use_fp16_court else 'FP32'}).")

    return ball_model, person_model, court_model, ball_local_model
