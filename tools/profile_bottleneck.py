"""
PROFILE BOTTLENECK - Tennis Analysis
Đo thời gian từng stage của pipeline (không sửa code pipeline: bọc hàm lúc chạy), chỉ ra stage nào
chiếm nhiều thời gian nhất, frame nào bị giật (spike) và vì stage nào.

Chạy (đặt cùng thư mục với main.py):
    python profile_bottleneck.py                  # video = config.INPUT_VIDEO
    python profile_bottleneck.py --video "D:\\a.mp4" --frames 400
    python profile_bottleneck.py --ask            # chọn video bằng hộp thoại
    python profile_bottleneck.py --no-sync        # không đồng bộ CUDA (thời gian GPU sẽ dồn vào stage sau)
    python profile_bottleneck.py --selftest       # kiểm tra chính công cụ, không cần model / GPU

Kết quả: bảng trong console + CSV từng frame + biểu đồ PNG trong REPORT_DIR.
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
# CÀI ĐẶT NGƯỜI DÙNG
# =====================================================================
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_report")
WARMUP_FRAMES = 20        # bỏ N frame đầu (khởi tạo CUDA / TensorRT / cudnn), báo riêng
MEASURE_FRAMES = 300      # số frame đo sau warmup
SYNC_CUDA = True          # đồng bộ CUDA quanh mỗi stage -> thời gian GPU tính đúng stage (chậm hơn chút)
WRITE_OUTPUT = True       # ghi video thật (để đo chi phí encode); file tạm, xoá sau khi đo
PREVIEW = False           # True = đo cả imshow (thường làm số đo nhiễu)
USE_CPROFILE = False      # True = thêm cProfile mức hàm Python (chậm hơn ~2x, chỉ để xem hotspot CPU)
SPIKE_FACTOR = 2.0        # frame có thời gian > N * median (hoặc > p95) là spike
SAVE_CSV = True
SAVE_PLOT = True          # cần matplotlib (không có thì bỏ qua)

# Gợi ý theo stage: (tiền tố nhãn, gợi ý). Khớp theo thứ tự.
ADVICE = [
    ("person.predict", "Model người: giảm PERSON_IMGSZ (1280 -> 960), dùng TensorRT (export engine + USE_TENSORRT_PERSON=True), "
                       "hoặc dùng model nhỏ hơn (yolo26l/m-pose); hoặc chạy person mỗi 2 frame và để PlayerMemory coast."),
    ("ball.predict[global]", "Quét toàn frame bằng model bóng rất tốn: tăng BALL_LOST_PATIENCE / BALL_LOCAL_MAX_MARGIN để ít rơi về global, "
                             "hoặc export engine TensorRT cho model bóng (USE_TENSORRT_BALL)."),
    ("ball.predict[local]", "Local search tốn: dùng engine local nhỏ (USE_BALL_LOCAL_ENGINE) hoặc giảm BALL_LOCAL_IMGSZ_MAX."),
    ("court.detect", "KeypointRCNN đồng bộ chặn main loop: tăng COURT_MIN_INTERVAL, dùng USE_FP16_COURT, hoặc giảm số lần mất track "
                     "(COURT_TRACK_MIN_VALID_RATIO)."),
    ("court.refine_kp", "refine_keypoint_by_lines (Canny + Hough x14 điểm): giảm REFINE_CROP_HALF hoặc bỏ refine khi detect nền."),
    ("court.validate", "is_valid_court (lấy mẫu pixel dọc cạnh): giảm số điểm mẫu (np.linspace 24 -> 12)."),
    ("court.optical_flow", "Optical flow: giảm OPTICAL_FLOW_SCALE (0.5 -> 0.35) hoặc LK_PARAMS maxLevel/winSize."),
    ("court.homography", "Tính homography mỗi lần set court: gọi ít hơn khi camera đứng yên."),
    ("court.update", "Logic court tracker (cvtColor + resize ảnh xám mỗi frame): đọc thêm phần 'court.*' bên dưới."),
    ("io.read_wait", "Decode video là cổ chai (main loop chờ reader): dùng decode GPU (NVDEC) / giảm độ phân giải / tăng IO_QUEUE_SIZE."),
    ("io.write_put", "Encode/ghi video là cổ chai (queue ghi đầy): dùng NVENC/ffmpeg thay mp4v, hoặc giảm độ phân giải đầu ra."),
    ("io.preview", "imshow + resize preview: tắt show_preview khi xử lý hàng loạt."),
    ("draw.", "Vẽ overlay: đơn giản hoá (tắt DEBUG_MODE, bỏ addWeighted mini-map nhiều lần)."),
    ("scene.", "SceneMonitor: histogram 160x90 mỗi frame - hiếm khi là cổ chai; nếu lớn kiểm tra cv2.resize / calcHist."),
    ("swing.", "Swing detector (dict + numpy nhỏ mỗi frame): nếu lớn, cache by_frame thay vì dựng lại mỗi frame."),
    ("bounce.", "Bounce detector: np.polyfit trong _slope / _trajectory_hit gọi mỗi frame - thay bằng công thức hồi quy tuyến tính trực tiếp."),
    ("person.candidates", "Gom candidate người: nhẹ; nếu lớn kiểm tra classify_player / project_point."),
    ("person.crop", "Crop quanh sân: nhẹ."),
    ("process_frame", "Phần còn lại trong process_frame (frame.copy, ...): copy ảnh 1080p mỗi frame; cân nhắc vẽ trực tiếp lên frame."),
]

GPU_LABELS = ("person.predict", "ball.predict", "court.detect")


# =====================================================================
# TIMER (phân cấp: self time = inclusive - thời gian các stage con)
# =====================================================================
class Timer:
    def __init__(self, sync=True):
        self.sync = bool(sync and torch is not None and torch.cuda.is_available())
        self.main_id = threading.get_ident()
        self.stack = []
        self.cur = None
        self.frames = []
        self.bg = defaultdict(lambda: [0, 0.0, 0.0])   # label -> [số lần, tổng ms, max ms] (thread nền)
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
    """Thay owner.name bằng bản bọc đo thời gian. label có thể là hàm (args, kwargs) -> str."""

    fn = getattr(owner, name)

    @functools.wraps(fn)
    def inner(*a, **k):
        with timer.section(label(a, k) if callable(label) else label):
            return fn(*a, **k)

    setattr(owner, name, inner)


# =====================================================================
# BÁO CÁO
# =====================================================================
def _group_of(label):
    if label.startswith("io."):
        return "IO (đọc/ghi/preview)"
    if any(label.startswith(g) for g in GPU_LABELS):
        return "GPU inference"
    return "CPU (logic / hình học / vẽ)"


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def analyze(frames, bg, info):
    """In báo cáo. Returns dict kết quả (dùng cho CSV / plot)."""

    n = len(frames)

    if n == 0:
        print("Không có frame nào để phân tích.")
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
    print(f"KẾT QUẢ ĐO  ({n} frame sau warmup)   video: {info.get('video', '?')}  "
          f"{info.get('w', '?')}x{info.get('h', '?')} @ {info.get('fps', 0):.1f}fps")
    print("=" * 100)
    print(f"Thời gian/frame: mean {mean_wall:.1f} ms | median {_pct(walls, 50):.1f} | p95 {_pct(walls, 95):.1f} "
          f"| p99 {_pct(walls, 99):.1f} | max {walls.max():.1f} ms")
    print(f"FPS: mean {fps_mean:.1f} | 1% low {1000.0 / max(_pct(walls, 99), 1e-6):.1f}"
          + (f" | video gốc {info['fps']:.1f}fps -> {'ĐỦ realtime' if fps_mean >= info['fps'] else 'CHƯA realtime'}"
             if info.get("fps") else ""))
    print(f"CUDA sync quanh stage: {'BẬT' if info.get('sync') else 'TẮT (thời gian GPU có thể dồn sang stage sau)'}")

    # ---- Bảng theo stage ----
    order = np.argsort(-S.mean(0))
    print("\n--- Thời gian THEO STAGE (self = trừ stage con; sắp theo self giảm dần) ---")
    print(f"{'stage':<26}{'gọi/frame':>10}{'self ms':>10}{'% frame':>9}{'p95':>9}{'max':>9}{'incl ms':>10}")

    for j in order:
        s = S[:, j]
        print(f"{labels[j]:<26}{C[:, j].mean():>10.2f}{s.mean():>10.2f}{100 * s.mean() / mean_wall:>8.1f}%"
              f"{_pct(s, 95):>9.1f}{s.max():>9.1f}{I[:, j].mean():>10.2f}")

    print(f"{'(chưa gán / overhead)':<26}{'':>10}{unacc.mean():>10.2f}{100 * unacc.mean() / mean_wall:>8.1f}%")

    # ---- Theo nhóm ----
    groups = defaultdict(float)

    for j, l in enumerate(labels):
        groups[_group_of(l)] += S[:, j].mean()

    print("\n--- Theo NHÓM ---")

    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{g:<30}{v:>8.1f} ms  {100 * v / mean_wall:>5.1f}%")

    # ---- Nếu bỏ stage X thì FPS bao nhiêu ----
    print("\n--- Nếu stage này nhanh gấp đôi / bỏ hẳn thì FPS ước tính ---")

    for j in order[:5]:
        m = S[:, j].mean()
        print(f"{labels[j]:<26} x2: {1000.0 / (mean_wall - m / 2):>6.1f} fps   bỏ hẳn: {1000.0 / max(mean_wall - m, 1e-6):>6.1f} fps")

    # ---- Spike ----
    thr = max(_pct(walls, 95), SPIKE_FACTOR * float(np.median(walls)))
    spike = walls >= thr

    if spike.sum() == 0:
        spike = walls >= _pct(walls, 95)

    if spike.sum() and (~spike).sum():
        diff = S[spike].mean(0) - S[~spike].mean(0)
        print(f"\n--- SPIKE: {int(spike.sum())} frame >= {thr:.0f} ms (bình thường ~{walls[~spike].mean():.0f} ms) ---")
        print("Stage gây spike (tăng thêm so với frame thường):")

        for j in np.argsort(-diff)[:5]:
            if diff[j] > 0.5:
                print(f"  {labels[j]:<26} +{diff[j]:>7.1f} ms")

        idx = np.flatnonzero(spike)

        if len(idx) >= 4:
            gaps = np.diff(idx)
            gaps = gaps[gaps > 1]

            if len(gaps) >= 3 and np.std(gaps) <= 0.25 * np.mean(gaps):
                print(f"  -> spike lặp theo chu kỳ ~{np.mean(gaps):.0f} frame (nghi detect nền / reset định kỳ)")

        print("5 frame chậm nhất:")

        for i in np.argsort(-walls)[:5]:
            top = np.argsort(-S[i])[:3]
            desc = ", ".join(f"{labels[j]} {S[i, j]:.0f}ms" for j in top)
            print(f"  frame đo #{i:<5} {walls[i]:>7.1f} ms  <- {desc}")

    # ---- Thread nền ----
    if bg:
        print("\n--- Thread nền (không chặn main loop, nhưng tranh GPU/CPU) ---")

        for l, (cnt, tot, mx) in bg.items():
            print(f"{l:<26} {cnt:>5} lần, mean {tot / cnt:>7.1f} ms, max {mx:>7.1f} ms")

    # ---- IO queue ----
    qr = [f["q_read"] for f in frames if "q_read" in f]
    qw = [f["q_write"] for f in frames if "q_write" in f]

    if qr:
        print(f"\nQueue đọc trung bình: {np.mean(qr):.1f}/{info.get('qsize', '?')} "
              f"(≈0: decode không kịp; đầy: GPU/CPU xử lý mới là cổ chai)")
        print(f"Queue ghi trung bình: {np.mean(qw):.1f}/{info.get('qsize', '?')} (đầy: encode là cổ chai)")

    if info.get("gpu_mem_mb"):
        print(f"VRAM đỉnh: {info['gpu_mem_mb']:.0f} MB")

    # ---- Kết luận ----
    top3 = order[:3]
    print("\n" + "=" * 100)
    print("KẾT LUẬN")
    print("=" * 100)
    gpu_share = sum(v for g, v in groups.items() if g.startswith("GPU")) / mean_wall
    io_share = sum(v for g, v in groups.items() if g.startswith("IO")) / mean_wall
    kind = "GPU" if gpu_share >= 0.5 else ("IO" if io_share >= 0.4 else "CPU / Python")
    print(f"Cổ chai chính: {labels[top3[0]]} ({100 * S[:, top3[0]].mean() / mean_wall:.0f}% thời gian frame). "
          f"Pipeline nghiêng về: {kind}-bound (GPU {100 * gpu_share:.0f}%, IO {100 * io_share:.0f}%).")

    g_calls = C[:, labels.index("ball.predict[global]")].mean() if "ball.predict[global]" in labels else 0.0
    l_calls = C[:, labels.index("ball.predict[local]")].mean() if "ball.predict[local]" in labels else 0.0

    if g_calls + l_calls > 0:
        print(f"Bóng: local {l_calls:.2f} / global {g_calls:.2f} lần predict mỗi frame "
              f"({100 * g_calls / (g_calls + l_calls):.0f}% là quét toàn frame).")

    print("\nGợi ý cho top 3 stage:")

    for j in top3:
        label = labels[j]
        tip = next((t for p, t in ADVICE if label.startswith(p)), "Chưa có gợi ý riêng cho stage này.")
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

        print(f"\n[OK] CSV từng frame: {path}")

    if SAVE_PLOT:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order = np.argsort(-S.mean(0))[:15][::-1]
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].barh([labels[j] for j in order], [S[:, j].mean() for j in order])
            ax[0].set_xlabel("ms / frame (self)")
            ax[0].set_title("Thời gian trung bình theo stage")
            ax[1].plot(walls, lw=0.8)
            ax[1].axhline(np.median(walls), color="g", ls="--", label="median")
            ax[1].set_xlabel("frame đo")
            ax[1].set_ylabel("ms")
            ax[1].set_title("Thời gian mỗi frame (spike = giật)")
            ax[1].legend()
            fig.tight_layout()
            path = os.path.join(REPORT_DIR, "bottleneck.png")
            fig.savefig(path, dpi=110)
            print(f"[OK] Biểu đồ: {path}")
        except Exception as e:  # noqa: BLE001
            print(f"[INFO] Bỏ qua biểu đồ ({e})")


# =====================================================================
# CHẠY PIPELINE THẬT
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

    # ---- bọc các stage ----
    def ball_label(a, k):
        src = a[0]
        return "ball.predict[global]" if src.shape[:2] == (H, W) else "ball.predict[local]"

    wrap_attr(court_tracker, "update", "court.update", timer)
    wrap_attr(CT, "track_court_by_optical_flow", "court.optical_flow", timer)
    wrap_attr(CT, "compute_court_homography", "court.homography", timer)
    wrap_attr(CT, "detect_court", "court.detect", timer)          # đồng bộ (main) hoặc nền (thread riêng)
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

    print(f"[INFO] Warmup {WARMUP_FRAMES} frame, đo {n_measure} frame ...")

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
                timer.bg.clear() if i == WARMUP_FRAMES - 1 else None   # thread nền của warmup không tính
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
        print(f"[INFO] Warmup: frame đầu {warm_walls[0]:.0f} ms (cold start), "
              f"trung bình warmup {np.mean(warm_walls):.0f} ms")

    info = {"video": os.path.basename(video), "w": W, "h": H, "fps": fps, "sync": timer.sync,
            "qsize": IO_QUEUE_SIZE}

    if torch is not None and torch.cuda.is_available():
        info["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6

    res = analyze(timer.frames, dict(timer.bg), info)
    save_outputs(res, timer.frames)

    if profiler is not None:
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("tottime").print_stats(25)
        print("\n--- cProfile (top 25 theo tottime, chỉ main thread) ---")
        print(s.getvalue())


# =====================================================================
# SELF-TEST (không cần model / GPU)
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

                if i % 20 == 0:                       # spike chu kỳ: detect đồng bộ
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
        title="Chọn video để đo", initialdir=start_dir if start_dir and os.path.isdir(start_dir) else None,
        filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("Tất cả", "*.*")])
    root.destroy()
    return path or None


def main():
    ap = argparse.ArgumentParser(description="Tìm bottleneck của pipeline Tennis Analysis")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=MEASURE_FRAMES, help="số frame đo sau warmup")
    ap.add_argument("--ask", action="store_true", help="chọn video bằng hộp thoại")
    ap.add_argument("--no-sync", action="store_true", help="không đồng bộ CUDA quanh stage")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    video = args.video or (pick_video_dialog(os.path.dirname(config.INPUT_VIDEO)) if args.ask else config.INPUT_VIDEO)

    if not video:
        print("[INFO] Chưa chọn video -> thoát.")
        return

    run_pipeline(video, args.frames, sync=SYNC_CUDA and not args.no_sync)


if __name__ == "__main__":
    main()
