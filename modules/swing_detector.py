"""Phát hiện cú vung vợt từ keypoint cổ tay của player (YOLO-pose)."""

from collections import deque

import numpy as np

from config import (
    KP_WRISTS, KP_TORSO, KP_TORSO_MIN, KP_MIN_CONF,
    SWING_SPEED_H_PER_S, SWING_MAX_SPEED_H_PER_S, SWING_MIN_FRAMES,
    SWING_BALL_DIST_H, SWING_TOL_S, SWING_SLOW_RATIO, SWING_RELEVANT_DIST_H,
    SWING_SPEED_SPAN, SWING_SHOW_S,
)


class SwingDetector:
    """
    Bằng chứng về cú vung vợt từ chuyển động cổ tay SO VỚI THÂN (tâm vai + hông):
      - camera pan / zoom, người chạy hay nhảy làm cả thân lẫn cổ tay dịch cùng nhau nên không tính là vung;
      - tốc độ tính bằng chiều cao người / giây (không phụ thuộc khoảng cách camera và fps),
        đo trên SWING_SPEED_SPAN frame để keypoint rung vài px không bị tính;
      - tốc độ phi thực tế (> SWING_MAX_SPEED_H_PER_S: keypoint nhảy, đảo trái/phải) bị bỏ;
      - phải nhanh liên tiếp >= SWING_MIN_FRAMES frame, nhiễu 1 frame không đủ.

    `swing_state` (dùng lọc bóng nảy) trả về 3 trạng thái:
        True   cổ tay vung nhanh VÀ sát bóng       -> chắc chắn là cú đánh
        False  cổ tay ở gần bóng, nhìn rõ, đang di chuyển chậm -> chắc chắn KHÔNG phải cú đánh
        None   không đủ thông tin (không thấy cổ tay / thân, hoặc tốc độ lưng chừng)

    `recent_swings` (dùng hiển thị) báo vung ngay ở frame hiện tại, không cần bóng.
    """

    def __init__(self, fps):
        self.fps = fps if fps and fps > 0 else 30.0
        self.tol = max(1, int(round(SWING_TOL_S * self.fps)))
        self.hist = {}      # half -> deque[(frame_idx, wrists_abs (2,2), wrists_rel (2,2), h)]
        self.streak = {}    # half -> (frame_idx cuối, số frame nhanh liên tiếp)

        # Hiển thị tín hiệu vung: half -> (frame_idx, toạ độ cổ tay, box player) của lần vung gần nhất
        self.show_frames = max(1, int(round(SWING_SHOW_S * self.fps)))
        self.last_swing = {}

    def reset(self):
        self.hist.clear()
        self.streak.clear()
        self.last_swing.clear()

    # ------------------------------------------------------------------ ghi nhận

    @staticmethod
    def _extract(kp):
        """(wrists_abs (2,2), wrists_rel (2,2)). rel = cổ tay - tâm thân; NaN nếu thiếu keypoint."""

        wr = np.full((2, 2), np.nan, dtype=np.float32)

        for j, k in enumerate(KP_WRISTS):
            if kp[k, 2] >= KP_MIN_CONF:
                wr[j] = kp[k, :2]

        torso = [kp[k, :2] for k in KP_TORSO if kp[k, 2] >= KP_MIN_CONF]

        if len(torso) >= KP_TORSO_MIN:
            rel = wr - np.mean(torso, axis=0).astype(np.float32)
        else:
            rel = np.full((2, 2), np.nan, dtype=np.float32)

        return wr, rel

    def update(self, frame_idx, players):
        """Gọi mỗi frame (kể cả khi không thấy bóng) để lịch sử cổ tay liên tục."""

        for p in players:
            kp = p.get("kpts")

            if kp is None:
                continue

            _, y1, _, y2 = p["box"]
            h = max(1.0, y2 - y1)

            wr, rel = self._extract(kp)

            dq = self.hist.setdefault(p["half"], deque(maxlen=int(self.fps) + 8))
            dq.append((frame_idx, wr, rel, h))

            self._mark_swing(p, dq, frame_idx, h)

    def _speeds_at(self, by_frame, f, h):
        """
        Tốc độ cổ tay so với thân tại frame f (chiều cao người / giây), shape (2,).
        NaN nếu thiếu mẫu / không thấy cổ tay hoặc thân / vượt SWING_MAX_SPEED_H_PER_S.
        """

        rel = by_frame[f][1]

        for span in range(SWING_SPEED_SPAN, 0, -1):   # thiếu mẫu thì thử khoảng ngắn hơn
            if f - span in by_frame:
                with np.errstate(invalid="ignore"):
                    speeds = np.linalg.norm(rel - by_frame[f - span][1], axis=1) / span / h * self.fps
                    speeds[speeds > SWING_MAX_SPEED_H_PER_S] = np.nan

                return speeds

        return np.full(2, np.nan, dtype=np.float32)

    def _mark_swing(self, player, dq, frame_idx, h):
        """Báo vung NGAY ở frame hiện tại (không chờ các frame sau, không cần bóng)."""

        by_frame = {f: (wr, rel) for f, wr, rel, _ in dq}
        speeds = self._speeds_at(by_frame, frame_idx, h)

        with np.errstate(invalid="ignore"):
            fast = speeds >= SWING_SPEED_H_PER_S

        half = player["half"]
        last_f, count = self.streak.get(half, (-10 ** 9, 0))

        if not np.any(fast):
            self.streak[half] = (frame_idx, 0)
            return

        count = count + 1 if frame_idx - last_f <= 2 else 1
        self.streak[half] = (frame_idx, count)

        if count >= SWING_MIN_FRAMES:
            j = int(np.nanargmax(speeds))   # cổ tay nhanh nhất = tay cầm vợt
            self.last_swing[half] = (frame_idx, by_frame[frame_idx][0][j].copy(), player["box"])

    # ------------------------------------------------------------------ truy vấn

    def recent_swings(self, frame_idx):
        """List (half, toạ độ cổ tay, box) của các cú vung còn trong thời gian hiển thị SWING_SHOW_S."""

        self.last_swing = {
            half: s for half, s in self.last_swing.items() if frame_idx - s[0] <= self.show_frames
        }

        return [(half, s[1], s[2]) for half, s in self.last_swing.items()]

    def _wrist_speeds(self, dq, frame_idx):
        """Với mỗi frame f quanh frame_idx: (f, wrists_abs, h, speeds (2,))."""

        by_frame = {f: (wr, rel) for f, wr, rel, _ in dq}
        heights = {f: h for f, _, _, h in dq}
        out = []

        for f in range(frame_idx - self.tol, frame_idx + self.tol + 1):
            if f in by_frame:
                out.append((f, by_frame[f][0], heights[f], self._speeds_at(by_frame, f, heights[f])))

        return out

    def _scan(self, frame_idx, ball_pt):
        """half -> (fast_near, seen_slow, seen_mid) quanh frame_idx. Xem swing_state."""

        ball = np.asarray(ball_pt, dtype=np.float32)
        out = {}

        for half, dq in self.hist.items():
            fast_near = 0   # số frame có cổ tay nhanh VÀ sát bóng
            seen_slow = 0   # số mẫu cổ tay GẦN bóng, nhìn rõ và chậm
            seen_mid = 0    # số mẫu gần bóng nhưng tốc độ lưng chừng

            for _, wr, h, speeds in self._wrist_speeds(dq, frame_idx):

                with np.errstate(invalid="ignore"):
                    dist = np.linalg.norm(wr - ball, axis=1) / h   # NaN -> mọi so sánh đều False

                    if np.any((speeds >= SWING_SPEED_H_PER_S) & (dist <= SWING_BALL_DIST_H)):
                        fast_near += 1

                    relevant = (dist <= SWING_RELEVANT_DIST_H) & ~np.isnan(speeds)

                    if np.any(relevant & (speeds < SWING_SPEED_H_PER_S * SWING_SLOW_RATIO)):
                        seen_slow += 1

                    if np.any(relevant & (speeds >= SWING_SPEED_H_PER_S * SWING_SLOW_RATIO)):
                        seen_mid += 1

            out[half] = (fast_near, seen_slow, seen_mid)

        return out

    def swing_state(self, frame_idx, ball_pt):
        """True / False / None, xem docstring của lớp."""

        scan = self._scan(frame_idx, ball_pt)

        fast_near = sum(v[0] for v in scan.values())
        seen_slow = sum(v[1] for v in scan.values())
        seen_mid = sum(v[2] for v in scan.values())

        if fast_near >= min(SWING_MIN_FRAMES, 2 * self.tol + 1):
            return True

        # Phải thấy chậm ở >= 2 mẫu và không có mẫu nào lưng chừng mới dám khẳng định "không vung"
        if seen_slow >= 2 and seen_mid == 0:
            return False

        return None

    def swing_hitter(self, frame_idx, ball_pt):
        """Half của người đang vung vợt sát bóng (cổ tay nhanh, gần bóng) tại frame_idx, hoặc None."""

        need = min(SWING_MIN_FRAMES, 2 * self.tol + 1)
        best = None

        for half, (fast_near, _, _) in self._scan(frame_idx, ball_pt).items():
            if fast_near >= need and (best is None or fast_near > best[1]):
                best = (half, fast_near)

        return best[0] if best else None

    def swing_near_ball(self, frame_idx, ball_pt):
        """Tương thích cũ: True nếu chắc chắn có cú vung sát bóng."""
        return self.swing_state(frame_idx, ball_pt) is True