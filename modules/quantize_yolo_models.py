import os
import yaml
from ultralytics import YOLO

import config
from extract_calibration_frames import extract_calibration_frames


def build_calibration_yaml(model, images_dir, yaml_path):
    """Tạo file dataset yaml tối thiểu cho bước calibration INT8, lấy
    đúng danh sách class name từ chính model (đảm bảo đúng số lớp nc,
    tránh lỗi mismatch khi model là bản train riêng như ball model)."""

    data = {
        "path": os.path.dirname(images_dir),
        "train": "images",
        "val": "images",
        "names": model.names,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    return yaml_path


def quantize_model(weights_path, engine_path, name, imgsz=640):

    print(f"\n=== Quantize {name} ({weights_path}) ===")

    model = YOLO(weights_path)

    images_dir = extract_calibration_frames()

    yaml_path = os.path.join(config.CALIBRATION_DATA_DIR, f"calib_{name}.yaml")
    build_calibration_yaml(model, images_dir, yaml_path)

    exported_path = model.export(
        format="engine",
        int8=True,
        data=yaml_path,
        imgsz=imgsz,
        dynamic=False,
        workspace=2,     
        device=0,
        half=False,        # int8=True đã bao hàm; half chỉ áp dụng khi
                            # không dùng int8
    )

    if os.path.abspath(str(exported_path)) != os.path.abspath(engine_path):
        print(
            f"CẢNH BÁO: engine build ra ở '{exported_path}', "
            f"khác với đường dẫn config kỳ vọng '{engine_path}'. "
            f"Hãy copy/đổi tên file cho khớp, hoặc sửa lại "
            f"config.BALL_MODEL_ENGINE_PATH / PERSON_MODEL_ENGINE_PATH."
        )

    print(f"Đã lưu engine: {exported_path}")

    return exported_path


def main():
    # quantize_model(config.BALL_MODEL_PATH, config.BALL_MODEL_ENGINE_PATH, "ball")
    quantize_model(config.PERSON_MODEL_PATH, config.PERSON_MODEL_ENGINE_PATH, "person")


if __name__ == "__main__":
    main()