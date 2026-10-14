"""
PROFILE BOTTLENECK - Tennis Analysis
Measures the time of each pipeline stage (does not modify the pipeline code: functions are wrapped at run time),
shows which stage takes the most time, which frames stutter (spikes) and because of which stage.

Run (place it in the same folder as main.py):
    python profile_bottleneck.py                  # video = config.INPUT_VIDEO
    python profile_bottleneck.py --video "path/to/video.mp4" --frames 400
    python profile_bottleneck.py --ask            # choose the video with a file dialog
    python profile_bottleneck.py --no-sync        # do not synchronize CUDA (GPU time will pile up in the next stage)
    python profile_bottleneck.py --selftest       # test the tool itself, no model / GPU needed

Output: tables in the console + a per-frame CSV + a PNG plot in REPORT_DIR.
"""

import argparse
import csv
import cProfile
import functools
import io
import os
import pstats
import sys
import tempfile
import threading
import time
from collections import defaultdict

import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

# =====================================================================
# USER SETTINGS
# =====================================================================
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_report")
WARMUP_FRAMES = 20        # skip the first N frames (CUDA / TensorRT / cudnn initialization), reported separately
MEASURE_FRAMES = 300      # number of frames measured after warmup
SYNC_CUDA = True          # synchronize CUDA around each stage -> GPU time is attributed to the right stage (slightly slower)
WRITE_OUTPUT = True       # write a real video (to measure encode cost); temporary file, deleted after measuring
PREVIEW = False           # True = also measure imshow (usually makes the numbers noisy)
USE_CPROFILE = False      # True = add function-level cProfile (about 2x slower, only to see CPU hotspots)
SPIKE_FACTOR = 2.0        # a frame taking > N * median (or > p95) is a spike
SAVE_CSV = True
SAVE_PLOT = True          # needs matplotlib (skipped if missing)

# Tips per stage: (label prefix, tip). Matched in order.
ADVICE = [
    ("person.predict", "Person model: lower PERSON_IMGSZ (1280 -> 960), use TensorRT (export the engine + USE_TENSORRT_PERSON=True), "
                       "or use a smaller model (yolo26l/m-pose); or run the person model every 2 frames and let PlayerMemory coast."),
    ("ball.predict[global]", "Full-frame scanning with the ball model is expensive: raise BALL_LOST_PATIENCE / BALL_LOCAL_MAX_MARGIN to fall back to global less often, "
                             "or export a TensorRT engine for the ball model (USE_TENSORRT_BALL)."),
    ("ball.predict[local]", "Local search is expensive: use the small local engine (USE_BALL_LOCAL_ENGINE) or lower BALL_LOCAL_IMGSZ_MAX."),
    ("court.detect", "Synchronous KeypointRCNN blocks the main loop: raise COURT_MIN_INTERVAL, use USE_FP16_COURT, or reduce how often tracking is lost "
                     "(COURT_TRACK_MIN_VALID_RATIO)."),
    ("court.refine_kp", "refine_keypoint_by_lines (Canny + Hough x14 points): lower REFINE_CROP_HALF or skip refinement during background detection."),
    ("court.validate", "is_valid_court (samples pixels along the edges): reduce the number of sample points (np.linspace 24 -> 12)."),
    ("court.optical_flow", "Optical flow: lower OPTICAL_FLOW_SCALE (0.5 -> 0.35) or LK_PARAMS maxLevel/winSize."),
    ("court.homography", "Homography is computed every time the court is set: call it less often when the camera is still."),
    ("court.update", "Court tracker logic (cvtColor + grayscale resize every frame): see the 'court.*' entries below."),
    ("io.read_wait", "Video decoding is the bottleneck (the main loop waits for the reader): use GPU decoding (NVDEC) / lower the resolution / raise IO_QUEUE_SIZE."),
    ("io.write_put", "Video encoding / writing is the bottleneck (the write queue is full): use NVENC/ffmpeg instead of mp4v, or lower the output resolution."),
    ("io.preview", "imshow + preview resize: turn off show_preview for batch processing."),
    ("draw.", "Overlay drawing: simplify (turn off DEBUG_MODE, avoid repeated mini-map addWeighted)."),
    ("scene.", "SceneMonitor: a 160x90 histogram every frame - rarely a bottleneck; if it is large, check cv2.resize / calcHist."),
    ("swing.", "Swing detector (small dicts + numpy every frame): if large, cache by_frame instead of rebuilding it every frame."),
    ("bounce.", "Bounce detector: np.polyfit in _slope / _trajectory_hit is called every frame - replace it with a direct linear regression formula."),
    ("person.candidates", "Person candidate grouping: light; if large, check classify_player / project_point."),
    ("person.crop", "Crop around the court: light."),
    ("process_frame", "The rest of process_frame (frame.copy, ...): copies a 1080p image every frame; consider drawing directly on the frame."),
]

GPU_LABELS = ("person.predict", "ball.predict", "court.detect")


# =====================================================================
# TIMER (hierarchical: self time = inclusive - time of child stages)
# =====================================================================
class Timer:
    def __init__(self, sync=True):
        self.sync = bool(sync and torch is not None and torch.cuda.is_available())
        self.main_id = threading.get_ident()
        self.stack = []
        self.cur = None
        self.frames = []
        self.bg = defaultdict(lambda: [0, 0.0, 0.0])   # label -> [count, total ms, max ms] (background threads)
        self._lock = threading.Lock()

    def _cuda_sync(self):
        if self.sync:
            torch.cuda.synchronize()

    def section(self, label):
        return _Section(self, label)

    def begin_frame(self):
        self.cur = {"self": defaultdict(float), "incl": defaultdict(float), "calls": defaultdict(int)}

    def end_frame(self, wall_s, extra=None):
        self.cur["wall"] = wall_s * 1e3
        self.cur.update(extra or {})
        self.frames.append(self.cur)
        self.cur = None


class _Section:
    __slots__ = ("tm", "label", "t0", "child", "main")

    def __init__(self, tm, label):
        self.tm, self.label = tm, label

    def __enter__(self):
        tm = self.tm
        self.main = threading.get_ident() == tm.main_id and tm.cur is not None

        if self.main:
            tm._cuda_sync()
            self.child = 0.0
            tm.stack.append(self)

        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        tm = self.tm

        if self.main:
            tm._cuda_sync()
            d = time.perf_counter() - self.t0
            tm.stack.pop()

            if tm.stack:
                tm.stack[-1].child += d

            f = tm.cur
            f["self"][self.label] += (d - self.child) * 1e3
            f["incl"][self.label] += d * 1e3
            f["calls"][self.label] += 1
        else:
            d = (time.perf_counter() - self.t0) * 1e3

            with tm._lock:
                rec = tm.bg[self.label]
                rec[0] += 1
                rec[1] += d
                rec[2] = max(rec[2], d)

        return False


def wrap_attr(owner, name, label, timer):
    """Replace owner.name with a timing wrapper. label may be a function (args, kwargs) -> str."""

    fn = getattr(owner, name)

    @functools.wraps(fn)
    def inner(*a, **k):
        with timer.section(label(a, k) if callable(label) else label):
            return fn(*a, **k)

    setattr(owner, name, inner)


# =====================================================================
# REPORT
# =====================================================================
def _group_of(label):
    if label.startswith("io."):
        return "IO (read/write/preview)"
    if any(label.startswith(g) for g in GPU_LABELS):
        return "GPU inference"
    return "CPU (logic / geometry / drawing)"


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def analyze(frames, bg, info):
    """Print the report. Returns a result dict (used for the CSV / plot)."""

    n = len(frames)

    if n == 0:
        print("No frames to analyze.")
        return {}

    labels = sorted({l for f in frames for l in f["self"]})
    walls = np.array([f["wall"] for f in frames])
    S = np.array([[f["self"].get(l, 0.0) for l in labels] for f in frames])
    I = np.array([[f["incl"].get(l, 0.0) for l in labels] for f in frames])
    C = np.array([[f["calls"].get(l, 0) for l in labels] for f in frames])

    mean_wall = float(walls.mean())
    fps_mean = 1000.0 / mean_wall
    unacc = np.maximum(walls - S.sum(1), 0.0)

    print("\n" + "=" * 100)
    print(f"RESULTS  ({n} frames after warmup)   video: {info.get('video', '?')}  "
          f"{info.get('w', '?')}x{info.get('h', '?')} @ {info.get('fps', 0):.1f}fps")
    print("=" * 100)
    print(f"Time/frame: mean {mean_wall:.1f} ms | median {_pct(walls, 50):.1f} | p95 {_pct(walls, 95):.1f} "
          f"| p99 {_pct(walls, 99):.1f} | max {walls.max():.1f} ms")
    print(f"FPS: mean {fps_mean:.1f} | 1% low {1000.0 / max(_pct(walls, 99), 1e-6):.1f}"
          + (f" | source video {info['fps']:.1f}fps -> {'REALTIME OK' if fps_mean >= info['fps'] else 'NOT realtime'}"
             if info.get("fps") else ""))
    print(f"CUDA sync around stages: {'ON' if info.get('sync') else 'OFF (GPU time may pile up in the next stage)'}")

    # ---- Per-stage table ----
    order = np.argsort(-S.mean(0))
    print("\n--- Time PER STAGE (self = excluding child stages; sorted by self, descending) ---")
    print(f"{'stage':<26}{'calls/fr':>10}{'self ms':>10}{'% frame':>9}{'p95':>9}{'max':>9}{'incl ms':>10}")

    for j in order:
        s = S[:, j]
        print(f"{labels[j]:<26}{C[:, j].mean():>10.2f}{s.mean():>10.2f}{100 * s.mean() / mean_wall:>8.1f}%"
              f"{_pct(s, 95):>9.1f}{s.max():>9.1f}{I[:, j].mean():>10.2f}")

    print(f"{'(unassigned / overhead)':<26}{'':>10}{unacc.mean():>10.2f}{100 * unacc.mean() / mean_wall:>8.1f}%")

    # ---- By group ----
    groups = defaultdict(float)

    for j, l in enumerate(labels):
        groups[_group_of(l)] += S[:, j].mean()

    print("\n--- By GROUP ---")

    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{g:<30}{v:>8.1f} ms  {100 * v / mean_wall:>5.1f}%")

    # ---- What would the FPS be without stage X ----
    print("\n--- Estimated FPS if this stage were twice as fast / removed entirely ---")

    for j in order[:5]:
        m = S[:, j].mean()
        print(f"{labels[j]:<26} x2: {1000.0 / (mean_wall - m / 2):>6.1f} fps   removed: {1000.0 / max(mean_wall - m, 1e-6):>6.1f} fps")

    # ---- Spike ----
    thr = max(_pct(walls, 95), SPIKE_FACTOR * float(np.median(walls)))
    spike = walls >= thr

    if spike.sum() == 0:
        spike = walls >= _pct(walls, 95)

    if spike.sum() and (~spike).sum():
        diff = S[spike].mean(0) - S[~spike].mean(0)
        print(f"\n--- SPIKE: {int(spike.sum())} frame >= {thr:.0f} ms (normally ~{walls[~spike].mean():.0f} ms) ---")
        print("Stages causing spikes (extra time versus a normal frame):")

        for j in np.argsort(-diff)[:5]:
            if diff[j] > 0.5:
                print(f"  {labels[j]:<26} +{diff[j]:>7.1f} ms")

        idx = np.flatnonzero(spike)

        if len(idx) >= 4:
            gaps = np.diff(idx)
            gaps = gaps[gaps > 1]

            if len(gaps) >= 3 and np.std(gaps) <= 0.25 * np.mean(gaps):
                print(f"  -> spikes repeat every ~{np.mean(gaps):.0f} frames (suspect background detection / periodic reset)")

        print("5 slowest frames:")

        for i in np.argsort(-walls)[:5]:
            top = np.argsort(-S[i])[:3]
            desc = ", ".join(f"{labels[j]} {S[i, j]:.0f}ms" for j in top)
            print(f"  measured frame #{i:<5} {walls[i]:>7.1f} ms  <- {desc}")

    # ---- Background threads ----
    if bg:
        print("\n--- Background threads (do not block the main loop, but compete for GPU/CPU) ---")

        for l, (cnt, tot, mx) in bg.items():
            print(f"{l:<26} {cnt:>5} calls, mean {tot / cnt:>7.1f} ms, max {mx:>7.1f} ms")

    # ---- IO queue ----
    qr = [f["q_read"] for f in frames if "q_read" in f]
    qw = [f["q_write"] for f in frames if "q_write" in f]

    if qr:
        print(f"\nAverage read queue: {np.mean(qr):.1f}/{info.get('qsize', '?')} "
              f"(≈0: decoding cannot keep up; full: GPU/CPU processing is the bottleneck)")
        print(f"Average write queue: {np.mean(qw):.1f}/{info.get('qsize', '?')} (full: encoding is the bottleneck)")

    if info.get("gpu_mem_mb"):
        print(f"Peak VRAM: {info['gpu_mem_mb']:.0f} MB")

    # ---- Conclusion ----
    top3 = order[:3]
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    gpu_share = sum(v for g, v in groups.items() if g.startswith("GPU")) / mean_wall
    io_share = sum(v for g, v in groups.items() if g.startswith("IO")) / mean_wall
    kind = "GPU" if gpu_share >= 0.5 else ("IO" if io_share >= 0.4 else "CPU / Python")
    print(f"Main bottleneck: {labels[top3[0]]} ({100 * S[:, top3[0]].mean() / mean_wall:.0f}% of frame time). "
          f"Pipeline is {kind}-bound (GPU {100 * gpu_share:.0f}%, IO {100 * io_share:.0f}%).")

    g_calls = C[:, labels.index("ball.predict[global]")].mean() if "ball.predict[global]" in labels else 0.0
    l_calls = C[:, labels.index("ball.predict[local]")].mean() if "ball.predict[local]" in labels else 0.0

    if g_calls + l_calls > 0:
        print(f"Ball: local {l_calls:.2f} / global {g_calls:.2f} predict calls per frame "
              f"({100 * g_calls / (g_calls + l_calls):.0f}% are full-frame scans).")

    print("\nTips for the top 3 stages:")

    for j in top3:
        label = labels[j]
        tip = next((t for p, t in ADVICE if label.startswith(p)), "No specific tip for this stage yet.")
        print(f"- {label} ({S[:, j].mean():.1f} ms): {tip}")

    return {"labels": labels, "S": S, "walls": walls, "mean_wall": mean_wall}


def save_outputs(res, frames):
    if not res:
        return

    os.makedirs(REPORT_DIR, exist_ok=True)
    labels, S, walls = res["labels"], res["S"], res["walls"]

    if SAVE_CSV:
        path = os.path.join(REPORT_DIR, "frame_times.csv")

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "wall_ms"] + labels)

            for i in range(len(walls)):
                w.writerow([i, f"{walls[i]:.2f}"] + [f"{v:.2f}" for v in S[i]])

        print(f"\n[OK] Per-frame CSV: {path}")

    if SAVE_PLOT:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order = np.argsort(-S.mean(0))[:15][::-1]
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].barh([labels[j] for j in order], [S[:, j].mean() for j in order])
            ax[0].set_xlabel("ms / frame (self)")
            ax[0].set_title("Mean time per stage")
            ax[1].plot(walls, lw=0.8)
            ax[1].axhline(np.median(walls), color="g", ls="--", label="median")
            ax[1].set_xlabel("measured frame")
            ax[1].set_ylabel("ms")
            ax[1].set_title("Time per frame (spike = stutter)")
            ax[1].legend()
            fig.tight_layout()
            path = os.path.join(REPORT_DIR, "bottleneck.png")
            fig.savefig(path, dpi=110)
            print(f"[OK] Plot: {path}")
        except Exception as e:  # noqa: BLE001
            print(f"[INFO] Skipping plot ({e})")


# =====================================================================
# RUN THE REAL PIPELINE
# =====================================================================
def run_pipeline(video, n_measure, sync):
    import cv2

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import ball_tracker as BT
    import court_detector as CD
    import court_tracker as CT
    import main as M
    from ball_tracker import BallTracker
    from bounce_detector import BounceDetector
    from config import IO_QUEUE_SIZE
    from court_tracker import CourtTracker
    from models import load_all_models
    from player_tracker import PlayerTracker
    from scene_cut import SceneMonitor
    from video_io import ThreadedVideoReader, ThreadedVideoWriter, open_video_io
    import player_tracker as PT

    t_load = time.perf_counter()
    ball_model, person_model, court_model, ball_local_model = load_all_models()
    print(f"[INFO] Load model: {time.perf_counter() - t_load:.1f}s")

    out_path = os.path.join(tempfile.gettempdir(), "profile_tmp_result.mp4")
    cap, out = open_video_io(video, out_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    court_tracker = CourtTracker(court_model)
    ball_tracker = BallTracker(ball_model, local_model=ball_local_model)
    person_tracker = PlayerTracker(person_model)
    scene_cut = SceneMonitor(fps)
    bounce = BounceDetector(fps, W)

    reader = ThreadedVideoReader(cap)
    writer = ThreadedVideoWriter(out) if WRITE_OUTPUT else None

    timer = Timer(sync=sync)

    # ---- wrap the stages ----
    def ball_label(a, k):
        src = a[0]
        return "ball.predict[global]" if src.shape[:2] == (H, W) else "ball.predict[local]"

    wrap_attr(court_tracker, "update", "court.update", timer)
    wrap_attr(CT, "track_court_by_optical_flow", "court.optical_flow", timer)
    wrap_attr(CT, "compute_court_homography", "court.homography", timer)
    wrap_attr(CT, "detect_court", "court.detect", timer)          # synchronous (main) or background (own thread)
    wrap_attr(CD, "refine_keypoint_by_lines", "court.refine_kp", timer)
    wrap_attr(CD, "is_valid_court", "court.validate", timer)

    wrap_attr(person_tracker, "update", "person.update", timer)
    wrap_attr(PT, "build_candidates", "person.candidates", timer)
    wrap_attr(PT, "_crop_around_court", "person.crop", timer)
    wrap_attr(person_model, "predict", "person.predict", timer)

    wrap_attr(ball_tracker, "detect", "ball.detect", timer)
    wrap_attr(ball_model, "predict", ball_label, timer)

    if ball_local_model is not None and ball_local_model is not ball_model:
        wrap_attr(ball_local_model, "predict", ball_label, timer)

    wrap_attr(bounce, "update", "bounce.update", timer)
    wrap_attr(bounce, "_trajectory_hit", "bounce.trajectory_hit", timer)
    wrap_attr(bounce, "_is_impulse", "bounce.impulse", timer)
    wrap_attr(bounce.swing, "update", "swing.update", timer)
    wrap_attr(bounce.swing, "swing_hitter", "swing.hitter", timer)
    wrap_attr(bounce.swing, "swing_state", "swing.state", timer)

    wrap_attr(scene_cut, "update", "scene.hist_cut", timer)
    wrap_attr(scene_cut, "update_signals", "scene.signals", timer)
    wrap_attr(scene_cut, "draw", "draw.scene_banner", timer)

    wrap_attr(M, "draw_court", "draw.court", timer)
    wrap_attr(M, "draw_players", "draw.players", timer)
    wrap_attr(M, "draw_mini_court", "draw.mini_map", timer)

    profiler = cProfile.Profile() if USE_CPROFILE else None
    total_frames = WARMUP_FRAMES + n_measure
    warm_walls = []

    print(f"[INFO] Warmup {WARMUP_FRAMES} frames, measuring {n_measure} frames ...")

    try:
        for i in range(total_frames):
            measuring = i >= WARMUP_FRAMES

            if measuring and profiler is not None and i == WARMUP_FRAMES:
                profiler.enable()

            timer.begin_frame()
            t0 = time.perf_counter()

            with timer.section("io.read_wait"):
                frame, aux = reader.read_aux()

            if frame is None:
                timer.cur = None
                break

            with timer.section("process_frame"):
                annotated = M.process_frame(
                    frame, court_tracker, ball_tracker, person_tracker, scene_cut, bounce, aux)

            with timer.section("overlay.fps_text"):
                cv2.putText(annotated, "FPS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if writer is not None:
                with timer.section("io.write_put"):
                    writer.write(annotated)

            if PREVIEW:
                with timer.section("io.preview"):
                    cv2.imshow("profile", cv2.resize(annotated, None, fx=0.5, fy=0.5))
                    cv2.waitKey(1)

            wall = time.perf_counter() - t0
            extra = {"q_read": reader.q.qsize(), "q_write": writer.q.qsize() if writer else 0}

            if measuring:
                timer.end_frame(wall, extra)
            else:
                warm_walls.append(wall * 1e3)
                timer.cur = None
                timer.bg.clear() if i == WARMUP_FRAMES - 1 else None   # warmup background threads are not counted
    finally:
        if profiler is not None:
            profiler.disable()

        reader.stop()

        if writer is not None:
            writer.close()

        cap.release()
        court_tracker.close()

        try:
            os.remove(out_path)
        except OSError:
            pass

    if warm_walls:
        print(f"[INFO] Warmup: first frame {warm_walls[0]:.0f} ms (cold start), "
              f"warmup average {np.mean(warm_walls):.0f} ms")

    info = {"video": os.path.basename(video), "w": W, "h": H, "fps": fps, "sync": timer.sync,
            "qsize": IO_QUEUE_SIZE}

    if torch is not None and torch.cuda.is_available():
        info["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6

    res = analyze(timer.frames, dict(timer.bg), info)
    save_outputs(res, timer.frames)

    if profiler is not None:
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("tottime").print_stats(25)
        print("\n--- cProfile (top 25 by tottime, main thread only) ---")
        print(s.getvalue())


# =====================================================================
# SELF-TEST (no model / GPU needed)
# =====================================================================
def selftest():
    tm = Timer(sync=False)

    def stage(label, ms):
        with tm.section(label):
            time.sleep(ms / 1000.0)

    def bg_job():
        with tm.section("court.detect"):
            time.sleep(0.03)

    for i in range(80):
        tm.begin_frame()
        t0 = time.perf_counter()
        stage("io.read_wait", 1)

        with tm.section("process_frame"):
            with tm.section("court.update"):
                stage("court.optical_flow", 3)

                if i % 20 == 0:                       # periodic spike: synchronous detect
                    stage("court.detect", 60)

                if i % 20 == 10:
                    threading.Thread(target=bg_job).start()

            stage("person.predict", 25)
            stage("ball.predict[local]", 6)
            stage("bounce.update", 2)
            stage("draw.mini_map", 1)

        stage("io.write_put", 2)
        tm.end_frame(time.perf_counter() - t0, {"q_read": 3, "q_write": 1})

    time.sleep(0.1)
    res = analyze(tm.frames, dict(tm.bg), {"video": "selftest", "w": 1920, "h": 1080, "fps": 30.0,
                                           "sync": False, "qsize": 16})
    save_outputs(res, tm.frames)


# =====================================================================
# MAIN
# =====================================================================
def pick_video_dialog(start_dir=None):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose a video to measure", initialdir=start_dir if start_dir and os.path.isdir(start_dir) else None,
        filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
    root.destroy()
    return path or None


def main():
    ap = argparse.ArgumentParser(description="Find the bottlenecks of the Tennis Analysis pipeline")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=MEASURE_FRAMES, help="number of frames to measure after warmup")
    ap.add_argument("--ask", action="store_true", help="choose the video with a file dialog")
    ap.add_argument("--no-sync", action="store_true", help="do not synchronize CUDA around stages")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    video = args.video or (pick_video_dialog(os.path.dirname(config.INPUT_VIDEO)) if args.ask else config.INPUT_VIDEO)

    if not video:
        print("[INFO] No video selected -> exiting.")
        return

    run_pipeline(video, args.frames, sync=SYNC_CUDA and not args.no_sync)


if __name__ == "__main__":
    main()
