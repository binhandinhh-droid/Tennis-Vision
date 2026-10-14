"""
Export YOLO models to TensorRT FP16 for the Tennis Analysis pipeline.

Creates up to 3 engines (FIXED shape, so imgsz at export time must match the one used at run time):

  person      PERSON_MODEL_PATH  -> <name>.engine           imgsz = PERSON_ENGINE_IMGSZ       (crop around the court)
  ball        BALL_MODEL_PATH    -> BALL_MODEL_ENGINE_PATH  imgsz = BALL_ENGINE_IMGSZ         (full-frame scan)
  ball-local  BALL_MODEL_PATH    -> BALL_LOCAL_ENGINE_PATH  imgsz = BALL_LOCAL_ENGINE_IMGSZ   (ROI around the ball)

Adjust the constants in the SETTINGS section below, then run from the project folder
(needs an NVIDIA GPU + `pip install tensorrt`):

    python export_engines.py

An engine is tied to the exact TensorRT version + GPU it was exported on: re-export after changing GPU / upgrading TensorRT.
"""

import os
import time

import numpy as np
from ultralytics import YOLO

import config as C

# ============================== SETTINGS ==============================

EXPORT_PERSON = True
EXPORT_BALL = False         # full-frame ball model
EXPORT_BALL_LOCAL = False    # ball model for the small ROI

FORCE = True                # True: export again even if the engine file already exists
HALF = True                 # True: FP16, False: FP32
VERIFY = True               # measure the speed of each engine after export
DEVICE = 0

PERSON_MODEL = C.PERSON_MODEL_PATH       # e.g. "yolo26m-pose.pt" for a smaller model (remember to update config to match)
PERSON_IMGSZ = C.PERSON_ENGINE_IMGSZ   # (h, w) or int
BALL_IMGSZ = C.BALL_ENGINE_IMGSZ       # (h, w) or int
BALL_LOCAL_IMGSZ = getattr(C, "BALL_LOCAL_ENGINE_IMGSZ", 192)   # the ROI is letterboxed to this size

# ======================================================================


def build_jobs():
    jobs = []

    if EXPORT_PERSON:
        jobs.append(dict(
            name="person", pt=PERSON_MODEL,
            dst=PERSON_MODEL.replace(".pt", ".engine"),   # same way config.py / models.py derive the path
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
    """Export one engine and move it to the right `dst` path. Returns True if a usable engine file exists."""

    name, pt, dst, imgsz = job["name"], job["pt"], job["dst"], job["imgsz"]

    sizes = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz,)

    if any(s % 32 for s in sizes):
        print(f"[{name}] WARNING: imgsz={imgsz} is not divisible by 32, ultralytics will round it.")

    if isinstance(imgsz, tuple):
        imgsz = list(imgsz)

    if os.path.exists(dst) and not FORCE:
        print(f"[{name}] {dst} already exists (skipped, set FORCE = True to export again)")
        return True

    print(f"[{name}] exporting {pt} -> {dst}  (imgsz={imgsz}, {'FP16' if HALF else 'FP32'}) ...")
    t0 = time.time()

    # ultralytics always writes to <.pt name>.engine. The global and local ball engines come from the same .pt
    # and therefore share the default output path: set the existing file aside so a later export does not overwrite an earlier engine.
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

    print(f"[{name}] done after {time.time() - t0:.0f}s: {dst}")
    return True


def verify(job, runs=30):
    """Load the engine and run a black image a few dozen times to measure predict time (incl. pre/post-processing)."""

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
        raise SystemExit("CUDA not found: TensorRT needs an NVIDIA GPU.")

    jobs = build_jobs()
    ok = []

    for job in jobs:
        try:
            if export_one(job):
                ok.append(job)
        except Exception as e:  # noqa: BLE001
            print(f"[{job['name']}] EXPORT ERROR: {e}")

    if VERIFY:
        print()

        for job in ok:
            try:
                verify(job)
            except Exception as e:  # noqa: BLE001
                print(f"[{job['name']}] VERIFY ERROR: {e}")

    print("\nNext, in config.py:")
    print("    USE_TENSORRT_PERSON = True")
    print("    USE_TENSORRT_BALL = True")
    print("    USE_BALL_LOCAL_ENGINE = True")
    print(f"    BALL_LOCAL_ENGINE_IMGSZ = {BALL_LOCAL_IMGSZ}")

    if PERSON_MODEL != C.PERSON_MODEL_PATH:
        print(f'    PERSON_MODEL_PATH = "{PERSON_MODEL}"')


if __name__ == "__main__":
    main()
