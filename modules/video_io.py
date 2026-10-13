"""Đọc / ghi video trên thread riêng, chạy song song với GPU."""

import queue
import threading

import cv2

from config import IO_QUEUE_SIZE, OPTICAL_FLOW_SCALE


def make_aux(frame):
    """
    Tiền xử lý CPU cho main loop, tính sẵn ở thread đọc (cv2 nhả GIL nên chạy song song với GPU).
    Kết quả GIỐNG HỆT lúc scene_cut / court_tracker tự tính:
        small: frame thu nhỏ 160x90 (SceneCutDetector)
        gray:  ảnh xám thu nhỏ theo OPTICAL_FLOW_SCALE (CourtTracker, optical flow)
    """

    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if OPTICAL_FLOW_SCALE != 1.0:
        gray = cv2.resize(
            gray, None, fx=OPTICAL_FLOW_SCALE, fy=OPTICAL_FLOW_SCALE,
            interpolation=cv2.INTER_AREA,
        )

    return {"small": small, "gray": gray}


class ThreadedVideoReader:
    """Đọc frame trên thread riêng, đẩy vào queue để chạy song song với GPU."""

    def __init__(self, cap, queue_size=IO_QUEUE_SIZE, precompute=True):
        self.cap = cap
        self.precompute = precompute
        self.q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        try:
            while not self._stop.is_set():
                ret, frame = self.cap.read()

                if not ret:
                    break

                aux = make_aux(frame) if self.precompute else {}

                if not self._put((frame, aux)):
                    break
        finally:
            self._put((None, None))   # sentinel: hết video

    def read_aux(self):
        """Trả về (frame, aux) tiếp theo; (None, None) khi hết video."""
        return self.q.get()

    def read(self):
        """Trả về frame tiếp theo, hoặc None khi hết video (không kèm aux)."""
        return self.q.get()[0]

    def stop(self):
        self._stop.set()
        self.thread.join()


class ThreadedVideoWriter:
    """Encode + ghi video trên thread riêng (mp4v chạy CPU khá tốn)."""

    def __init__(self, writer, queue_size=IO_QUEUE_SIZE):
        self.writer = writer
        self.q = queue.Queue(maxsize=queue_size)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            frame = self.q.get()

            if frame is None:
                break

            if self.error is not None:
                continue   # vẫn rút queue để main thread không bị kẹt

            try:
                self.writer.write(frame)
            except Exception as e:  # noqa: BLE001
                self.error = e
                print(f"[Writer] lỗi ghi video: {e}")

    def write(self, frame):
        self.q.put(frame)

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.writer.release()


def open_video_io(input_video, output_video):

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {input_video}")

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), size)

    return cap, out
