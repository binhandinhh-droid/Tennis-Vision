"""
Export model YOLO sang TensorRT FP16 cho pipeline Tennis Analysis.

Tạo tối đa 3 engine (shape CỐ ĐỊNH, nên imgsz lúc export phải đúng với lúc chạy):

  person      PERSON_MODEL_PATH  -> <tên>.engine           imgsz = PERSON_IMGSZ       (crop quanh sân)
  ball        BALL_MODEL_PATH    -> BALL_MODEL_ENGINE_PATH imgsz = BALL_TRAIN_IMGSZ   (quét toàn frame)
  ball-local  BALL_MODEL_PATH    -> BALL_LOCAL_ENGINE_PATH imgsz = BALL_LOCAL_ENGINE_IMGSZ (ROI quanh bóng)

Chỉnh các hằng số ở phần CẤU HÌNH bên dưới rồi chạy từ thư mục project
(cần GPU NVIDIA + `pip install tensorrt`):

    python export_engines.py

Engine gắn với đúng phiên bản TensorRT + GPU đã export: đổi GPU / nâng TensorRT thì export lại.
"""

import os
import time

import numpy as np
from ultralytics import YOLO

import config as C

# ============================== CẤU HÌNH ==============================

EXPORT_PERSON = True
EXPORT_BALL = False         # model bóng quét toàn frame
EXPORT_BALL_LOCAL = False    # model bóng cho ROI nhỏ

FORCE = True                # True: export lại dù file engine đã tồn tại
HALF = True                 # True: FP16, False: FP32
VERIFY = True               # đo tốc độ từng engine sau khi export
DEVICE = 0

PERSON_MODEL = C.PERSON_MODEL_PATH       # vd "yolo26m-pose.pt" nếu muốn model nhỏ hơn (nhớ sửa config cho khớp)
PERSON_IMGSZ = C.PERSON_ENGINE_IMGSZ   # (h, w) hoặc int
BALL_IMGSZ = C.BALL_ENGINE_IMGSZ       # (h, w) hoặc int
BALL_LOCAL_IMGSZ = getattr(C, "BALL_LOCAL_ENGINE_IMGSZ", 192)   # ROI sẽ bị letterbox về cỡ này

# ======================================================================


def build_jobs():
    jobs = []

    if EXPORT_PERSON:
        jobs.append(dict(
            name="person", pt=PERSON_MODEL,
            dst=PERSON_MODEL.replace(".pt", ".engine"),   # giống cách config.py / models.py suy ra đường dẫn
            imgsz=PERSON_IMGSZ, test_shape=(1080, 1920),
        ))

    if EXPORT_BALL:
        jobs.append(dict(
            name="ball", pt=C.BALL_MODEL_PATH, dst=C.BALL_MODEL_ENGINE_PATH,
            imgsz=BALL_IMGSZ, test_shape=(1080, 1920),
        ))

    if EXPORT_BALL_LOCAL:
        jobs.append(dict(
            name="ball-local", pt=C.BALL_MODEL_PATH, dst=C.BALL_LOCAL_ENGINE_PATH,
            imgsz=BALL_LOCAL_IMGSZ, test_shape=(240, 240),
        ))

    return jobs


def export_one(job):
    """Export 1 engine rồi đưa về đúng đường dẫn `dst`. Returns True nếu có file engine dùng được."""

    name, pt, dst, imgsz = job["name"], job["pt"], job["dst"], job["imgsz"]

    sizes = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz,)

    if any(s % 32 for s in sizes):
        print(f"[{name}] CẢNH BÁO: imgsz={imgsz} không chia hết cho 32, ultralytics sẽ tự làm tròn.")

    if isinstance(imgsz, tuple):
        imgsz = list(imgsz)

    if os.path.exists(dst) and not FORCE:
        print(f"[{name}] đã có {dst} (bỏ qua, đặt FORCE = True để export lại)")
        return True

    print(f"[{name}] export {pt} -> {dst}  (imgsz={imgsz}, {'FP16' if HALF else 'FP32'}) ...")
    t0 = time.time()

    # ultralytics luôn ghi ra <tên .pt>.engine. Ball global và ball local cùng đến từ 1 file .pt
    # nên cùng đường ra mặc định: cất file có sẵn đi để export sau không đè mất engine trước.
    default_out = os.path.splitext(os.path.abspath(pt))[0] + ".engine"
    backup = None

    if default_out != os.path.abspath(dst) and os.path.exists(default_out):
        backup = default_out + ".bak"
        os.replace(default_out, backup)

    try:
        model = YOLO(pt)
        out = model.export(
            format="engine", imgsz=imgsz, half=HALF, batch=1, device=DEVICE,
            dynamic=False, simplify=True, verbose=False,
        )
        out = os.path.abspath(str(out))

        if out != os.path.abspath(dst):
            os.replace(out, dst)
    finally:
        if backup is not None:
            os.replace(backup, default_out)

    print(f"[{name}] xong sau {time.time() - t0:.0f}s: {dst}")
    return True


def verify(job, runs=30):
    """Nạp engine, chạy ảnh đen vài chục lần để đo thời gian predict (gồm tiền/hậu xử lý)."""

    model = YOLO(job["dst"])
    img = np.zeros((*job["test_shape"], 3), np.uint8)

    for _ in range(5):   # warmup
        model.predict(img, verbose=False)

    times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(img, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)

    print(
        f"[{job['name']}] verify {job['test_shape'][1]}x{job['test_shape'][0]}: "
        f"mean {np.mean(times):.1f} ms | p95 {np.percentile(times, 95):.1f} ms"
    )


def main():
    if C.DEVICE.type != "cuda":
        raise SystemExit("Không thấy CUDA: TensorRT cần GPU NVIDIA.")

    jobs = build_jobs()
    ok = []

    for job in jobs:
        try:
            if export_one(job):
                ok.append(job)
        except Exception as e:  # noqa: BLE001
            print(f"[{job['name']}] LỖI export: {e}")

    if VERIFY:
        print()

        for job in ok:
            try:
                verify(job)
            except Exception as e:  # noqa: BLE001
                print(f"[{job['name']}] LỖI verify: {e}")

    print("\nTiếp theo, trong config.py:")
    print("    USE_TENSORRT_PERSON = True")
    print("    USE_TENSORRT_BALL = True")
    print("    USE_BALL_LOCAL_ENGINE = True")
    print(f"    BALL_LOCAL_ENGINE_IMGSZ = {BALL_LOCAL_IMGSZ}")

    if PERSON_MODEL != C.PERSON_MODEL_PATH:
        print(f'    PERSON_MODEL_PATH = "{PERSON_MODEL}"')


if __name__ == "__main__":
    main()
