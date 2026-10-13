"""Phát hiện điểm bóng nảy trên sân từ quỹ đạo bóng, trả về toạ độ trên mini-map."""

from collections import deque

import numpy as np

from config import (
    DEBUG_MODE, MINI_COURT_W, MINI_COURT_H,
    BALL_EVENT_MAX_GAP_S, BALL_BOUNCE_COOLDOWN_S, BOUNCE_HALF_WINDOW_S, BOUNCE_MIN_DROP_PX,
    BOUNCE_HIT_SPEED_GAIN, BALL_AIR_MIN_RAISE, HIT_REACH_X, HIT_REACH_UP,
    BOUNCE_MAX_OUTSIDE_PX, BOUNCE_MARK_S, BOUNCE_MIN_DVY_PX,
    HIT_SIDE_EXCL_S, HIT_AFTER_TOL_S, HIT_EVENT_KEEP_S,
    SWING_SHOW_S,
    HIT_TRAJ_ENABLE, HIT_TRAJ_MIN_SPEED_PX, HIT_TRAJ_TURN_DEG, HIT_TRAJ_SPEED_RATIO,
    HIT_TRAJ_GROUND_RATIO, HIT_TRAJ_APPROACH_H, HIT_DUP_S,
)
from geometry import project_point
from swing_detector import SwingDetector


def _half_of(y):
    """Nửa sân chứa điểm mini-map có toạ độ y. Cùng nhãn với player đứng ở nửa đó."""
    return "Player 2" if y < MINI_COURT_H / 2.0 else "Player 1"


class BounceDetector:
    """
    Điểm nảy = đáy quỹ đạo (hoặc xung giảm vy) của cửa sổ 2k+1 mẫu bóng liên tiếp.
    Sau đó ứng viên phải qua các luật loại "cú đánh" (kết quả trễ k mẫu so với frame thật):

      L1  Trong tầm vợt, cao hơn chân: keypoint cổ tay / quỹ đạo quyết định (như cũ).
      L2  Sổ cú đánh (hit events): mỗi khi cổ tay nhanh sát bóng (hoặc L1 kết luận cú đánh)
          ghi lại (frame, nửa sân người đánh, vị trí chân). Rồi:
            - ứng viên nảy ở NỬA SÂN CỦA CHÍNH NGƯỜI VỪA ĐÁNH, trong HIT_SIDE_EXCL_S sau cú đánh
              -> loại (cú đánh gần như luôn sang sân đối thủ)
    Ngưỡng px tính ở bề rộng 1920 và tự scale theo độ phân giải video.
    """

    def __init__(self, fps, frame_width):
        fps = fps if fps and fps > 0 else 30.0

        self.fps = fps
        self.scale = frame_width / 1920.0
        self.max_gap = max(2, int(round(BALL_EVENT_MAX_GAP_S * fps)))
        self.cooldown = max(2, int(round(BALL_BOUNCE_COOLDOWN_S * fps)))
        self.mark_frames = max(1, int(round(BOUNCE_MARK_S * fps)))
        self.half = max(2, int(round(BOUNCE_HALF_WINDOW_S * fps)))

        self.hit_side_excl = max(2, int(round(HIT_SIDE_EXCL_S * fps)))
        self.hit_after_tol = max(1, int(round(HIT_AFTER_TOL_S * fps)))
        self.hit_dup = max(3, int(round(HIT_DUP_S * fps)))
        self.show_frames = max(1, int(round(SWING_SHOW_S * fps)))
        self.hit_keep = max(2, int(round(HIT_EVENT_KEEP_S * fps)))

        self.frame_idx = 0
        self._last_bounce = -10 ** 9
        self.samples = deque(maxlen=2 * self.half + 1)   # (frame_idx, pt, proj, players, feet)
        self.marks = []                                  # [(frame_idx, x, y)] toạ độ mini-map
        self.swing = SwingDetector(fps)

        self.shown = {}            # half -> frame_idx lần báo HIT gần nhất (để hiển thị)
        self.hits = []             # [[frame_đầu, half người đánh, chân (mini-map) hoặc None, frame_cuối]]

    def reset(self):
        """Chuyển cảnh: bỏ quỹ đạo, điểm nảy và sổ cú đánh cũ."""
        self.samples.clear()
        self.marks.clear()
        self.swing.reset()
        self.hits.clear()
        self.shown.clear()

    # ------------------------------------------------------------------ sổ cú đánh

    @staticmethod
    def _project_feet(H, players):
        feet = {}

        for p in players:
            pr = project_point(H, p["point"])

            if pr is not None:
                feet[p["half"]] = pr

        return feet

    def _register_hit(self, frame, half, feet):
        """Ghi 1 cú đánh (từ cổ tay, quỹ đạo hoặc suy luận). Tín hiệu trùng cú cũ thì gộp lại."""

        xy = feet.get(half)
        self.shown[half] = self.frame_idx

        for h in self.hits:
            if h[1] == half and h[0] - self.hit_dup <= frame <= h[3] + self.hit_dup:
                h[0] = min(h[0], frame)
                h[3] = max(h[3], frame)

                if h[2] is None:
                    h[2] = xy

                return

        self.hits.append([frame, half, xy, frame])
        self.hits = [h for h in self.hits if frame - h[3] <= self.hit_keep]

        if DEBUG_MODE:
            print(f"[Hit] frame {frame}: {half} đánh bóng")

    def recent_hit_halves(self):
        """Tập half đang trong thời gian hiển thị HIT (từ cổ tay HOẶC quỹ đạo bóng)."""

        self.shown = {h: f for h, f in self.shown.items() if self.frame_idx - f <= self.show_frames}

        return set(self.shown)

    # ---- phát hiện cú đánh từ quỹ đạo ----

    @staticmethod
    def _vel(frames, pts, a, b):
        """Vận tốc (px/frame) của các mẫu a..b: độ dốc hồi quy tuyến tính theo frame."""
        return np.array([np.polyfit(frames[a:b + 1], pts[a:b + 1, k], 1)[0] for k in (0, 1)])

    @staticmethod
    def _player_of(players, half):
        """(cx, cy, h, y2) của player ở half, hoặc None."""
        for p in players:
            if p["half"] == half:
                x1, y1, x2, y2 = p["box"]
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1), float(y2)
        return None

    @staticmethod
    def _reach_half(pt, players):
        """Half của player gần nhất mà bóng đang trong tầm vợt (không đòi bóng cao hơn chân)."""

        bx, by = float(pt[0]), float(pt[1])
        best, best_d = None, 1e18

        for p in players:
            x1, y1, x2, y2 = p["box"]
            h = max(1.0, y2 - y1)

            if (x1 - HIT_REACH_X * h <= bx <= x2 + HIT_REACH_X * h
                    and y1 - HIT_REACH_UP * h <= by <= y2 + 0.1 * h):
                d = np.hypot((x1 + x2) / 2.0 - bx, (y1 + y2) / 2.0 - by)

                if d < best_d:
                    best, best_d = p["half"], d

        return best

    def _trajectory_hit(self, fb, pb, frames, pts, players_seq, feet_b):
        """
        Cú đánh từ quỹ đạo: bóng trong tầm vợt, hướng đi quay ngược lại (>= HIT_TRAJ_TURN_DEG),
        tiến sát rồi rời xa player, và không mất tốc (bóng nảy mới mất tốc). Cổ tay đã khẳng
        định "chậm, không vung" (swing_state False) thì bỏ qua.
        """

        if not HIT_TRAJ_ENABLE:
            return

        c = self.half
        v_in = self._vel(frames, pts, 0, c)
        v_out = self._vel(frames, pts, c, len(frames) - 1)
        s_in, s_out = float(np.hypot(*v_in)), float(np.hypot(*v_out))

        if min(s_in, s_out) < HIT_TRAJ_MIN_SPEED_PX * self.scale:
            return

        if float(v_in @ v_out) / (s_in * s_out) > np.cos(np.radians(HIT_TRAJ_TURN_DEG)):
            return   # chưa quay ngược đủ lớn

        players_b = players_seq[c]
        half = self._reach_half(pb, players_b)

        if half is None:
            return

        cx, cy, h, y2 = self._player_of(players_b, half)

        on_ground = pb[1] >= y2 - BALL_AIR_MIN_RAISE * h
        need = HIT_TRAJ_GROUND_RATIO if on_ground else HIT_TRAJ_SPEED_RATIO

        if s_out / s_in < need:
            return   # mất tốc: giống nảy hơn là đánh

        # Tiến sát rồi rời xa chính player đó
        p0 = self._player_of(players_seq[0], half)
        p1 = self._player_of(players_seq[-1], half)

        if p0 is not None and p1 is not None:
            d_b = np.hypot(pb[0] - cx, pb[1] - cy) / h
            d_0 = np.hypot(pts[0][0] - p0[0], pts[0][1] - p0[1]) / p0[2]
            d_1 = np.hypot(pts[-1][0] - p1[0], pts[-1][1] - p1[1]) / p1[2]

            if d_0 - d_b < HIT_TRAJ_APPROACH_H or d_1 - d_b < HIT_TRAJ_APPROACH_H:
                return

        if self.swing.swing_state(fb, pb) is False:
            return

        if DEBUG_MODE:
            print(f"[Hit] frame {fb}: quỹ đạo quay ngược ({half}), tốc độ sau/trước = {s_out / s_in:.2f}")

        self._register_hit(fb, half, feet_b)

    def _hit_context_reject(self, fb, half_b):
        """Trả về lý do loại (str) nếu ứng viên nảy thực ra là hệ quả của 1 cú đánh, ngược lại None."""

        # L2: theo sổ cú đánh (chỉ loại theo nửa sân; luật "sát chân" đã bỏ vì chồng với luật này)
        for first, half, _, _ in self.hits:
            dt = fb - first

            if dt < -self.hit_after_tol:
                continue   # cú đánh xảy ra SAU lần nảy này (nảy rồi mới đánh): không liên quan

            if dt <= self.hit_side_excl and half == half_b:
                return f"nảy ở nửa sân của {half} ngay sau cú đánh của người đó"

        return None

    # ------------------------------------------------------------------ update

    def update(self, ball_point, players, H):
        """Gọi mỗi frame. Returns (x, y) điểm nảy vừa phát hiện (toạ độ mini-map) hoặc None."""

        self.frame_idx += 1

        self.swing.update(self.frame_idx, players)   # ghi nhận cổ tay mỗi frame, trước các return sớm

        if ball_point is None or H is None:
            return None

        proj = project_point(H, ball_point)

        if proj is None:
            return None

        feet = self._project_feet(H, players)

        # Cổ tay vung nhanh sát bóng ở frame hiện tại -> ghi nhận cú đánh (không phụ thuộc quỹ đạo)
        hitter = self.swing.swing_hitter(self.frame_idx, ball_point)

        if hitter is not None:
            self._register_hit(self.frame_idx, hitter, feet)

        if self.samples and self.frame_idx - self.samples[-1][0] > self.max_gap:
            self.samples.clear()   # mất bóng quá lâu: quỹ đạo cũ không còn liên tục

        self.samples.append(
            (self.frame_idx, np.array(ball_point, dtype=np.float32), proj, players, feet))

        if len(self.samples) < self.samples.maxlen:
            return None

        c = self.half
        frames = [s[0] for s in self.samples]
        pts = np.array([s[1] for s in self.samples])
        ys = pts[:, 1]

        fb, pb, proj_b, players_b, feet_b = self.samples[c]

        # Cú đánh từ quỹ đạo: chạy mỗi frame, độc lập với việc có ứng viên nảy hay không
        self._trajectory_hit(fb, pb, frames, pts, [s[3] for s in self.samples], feet_b)

        min_drop = BOUNCE_MIN_DROP_PX * self.scale

        # Đường 1: mẫu giữa là đáy quỹ đạo (y ảnh lớn nhất) của cả cửa sổ
        is_bottom = (int(np.argmax(ys)) == c
                     and ys[c] - ys[0] >= min_drop and ys[c] - ys[-1] >= min_drop)

        # Đường 2: vy giảm đột ngột dù y ảnh không đổi chiều
        if not is_bottom and not self._is_impulse(frames, ys):
            return None

        if fb - self._last_bounce < self.cooldown:
            return None

        v_in = (pb - pts[0]) / (fb - frames[0])
        v_out = (pts[-1] - pb) / (frames[-1] - fb)

        # L1: bóng trong tầm vợt VÀ cao hơn chân
        if self._in_racket_reach(pb, players_b):
            swing = self.swing.swing_state(fb, pb)
            hit_like = self._hit_like(v_in, v_out)

            if DEBUG_MODE:
                print(f"[Bounce] frame {fb}: trong tầm vợt, swing={swing}, hit_like={hit_like}")

            if swing is True or (swing is None and hit_like):
                half_hit = self._reach_half(pb, players_b)

                if half_hit is not None:
                    self._register_hit(fb, half_hit, feet_b)

                return self._reject("cú đánh trong tầm vợt", fb)

        x, y = proj_b
        half_b = _half_of(y)

        # L2: ngữ cảnh cú đánh
        reason = self._hit_context_reject(fb, half_b)

        if reason is not None:
            return self._reject(reason, fb)

        if not (-BOUNCE_MAX_OUTSIDE_PX <= x <= MINI_COURT_W + BOUNCE_MAX_OUTSIDE_PX
                and -BOUNCE_MAX_OUTSIDE_PX <= y <= MINI_COURT_H + BOUNCE_MAX_OUTSIDE_PX):
            return self._reject("rơi quá xa sân", fb)

        self._last_bounce = fb
        self.marks.append((fb, float(x), float(y)))

        if DEBUG_MODE:
            print(f"[Bounce] frame {fb}: mini-map ({x:.0f}, {y:.0f}) nửa sân {half_b}")

        return float(x), float(y)

    # ------------------------------------------------------------------ tiện ích

    @staticmethod
    def _slope(frames, ys, a, b):
        return float(np.polyfit(frames[a:b + 1], ys[a:b + 1], 1)[0])

    def _is_impulse(self, frames, ys):
        """Bóng đang đi xuống rồi vy giảm đột ngột (kể cả khi y ảnh không đổi chiều)."""
        c = self.half
        m = min(3, self.half - 1)

        if m < 1:
            return False

        def dvy(i):
            vin = self._slope(frames, ys, i - m, i)
            vout = self._slope(frames, ys, i, i + m)
            return vout - vin, vin

        vals = [dvy(i) for i in (c - 1, c, c + 1)]
        d, vin = vals[1]

        # Nảy = vy bị giật ngược lên (d rất âm). KHÔNG đòi vin > 0: bóng nảy xiên bay ra xa camera
        # đang đi lên trong ảnh (vin <= 0) trước khi nảy, y ảnh không có "đáy".
        if d > -BOUNCE_MIN_DVY_PX * self.scale:
            return False

        if DEBUG_MODE:
            print(f"[Bounce] xung vy: vin={vin:.1f} d={d:.1f}")

        return d <= min(v[0] for v in vals)

    def _reject(self, reason, frame):
        if DEBUG_MODE:
            print(f"[Bounce] frame {frame}: bỏ ({reason})")
        return None

    def _hit_like(self, v_in, v_out):
        """Cú đánh: bóng tăng tốc rõ rệt hoặc đảo chiều ngang."""

        speed_in = float(np.hypot(*v_in))
        speed_out = float(np.hypot(*v_out))

        if speed_in > 1e-6 and speed_out / speed_in > BOUNCE_HIT_SPEED_GAIN:
            return True

        vx_min = 2.0 * self.scale
        return v_in[0] * v_out[0] < 0 and abs(v_in[0]) > vx_min and abs(v_out[0]) > vx_min

    def recent_swings(self):
        """List (half, toạ độ cổ tay, box) các cú vung vợt còn trong thời gian hiển thị."""
        return self.swing.recent_swings(self.frame_idx)

    def recent_marks(self):
        """List (x, y) các điểm nảy còn trong thời gian hiển thị (toạ độ mini-map)."""

        self.marks = [m for m in self.marks if self.frame_idx - m[0] <= self.mark_frames]

        return [(x, y) for _, x, y in self.marks]

    @staticmethod
    def _in_racket_reach(pt, players):
        """True nếu bóng đang trong tầm vợt của 1 player và cao hơn chân quá BALL_AIR_MIN_RAISE."""

        bx, by = float(pt[0]), float(pt[1])

        for p in players:
            x1, y1, x2, y2 = p["box"]
            h = max(1.0, y2 - y1)

            in_reach = (x1 - HIT_REACH_X * h <= bx <= x2 + HIT_REACH_X * h
                        and y1 - HIT_REACH_UP * h <= by <= y2)

            if in_reach and by <= y2 - BALL_AIR_MIN_RAISE * h:
                return True

        return False