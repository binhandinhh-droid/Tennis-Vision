import torch
import config
from models import build_court_model


def quantize_court_model_fp16(
    weights_path=config.COURT_MODEL_PATH,
    output_path=config.COURT_MODEL_FP16_PATH,
    num_keypoints=config.NUM_KEYPOINTS,
):
    print(f"Load FP32 weights: {weights_path}")

    model = build_court_model(num_keypoints)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    model.to(config.DEVICE)
    model.half()
    model.eval()

    dummy = torch.randn(3, 720, 1280, dtype=torch.float16, device=config.DEVICE)

    try:
        with torch.no_grad():
            _ = model([dummy])
        print("Sanity check FP16 forward pass: OK")
    except Exception as e:
        print("LỖI: forward pass FP16 thất bại, KHÔNG lưu bản FP16.")
        print("Giữ nguyên dùng bản FP32 gốc cho court model.")
        print(f"Chi tiết lỗi: {e}")
        return None

    torch.save(model.state_dict(), output_path)
    print(f"Đã lưu weights FP16: {output_path}")

    return output_path


if __name__ == "__main__":
    quantize_court_model_fp16()
