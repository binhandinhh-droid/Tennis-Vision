"""Detect ball bounce points on the court from the ball trajectory; returns mini-map coordinates."""

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
    """Court half containing the mini-map point with coordinate y. Same label as the player standing in that half."""
    return "Player 2" if y < MINI_COURT_H / 2.0 else "Player 1"


class BounceDetector:
    """
    Bounce point = trajectory bottom (or vy-drop impulse) of a window of 2k+1 consecutive ball samples.
    The candidate then has to pass the rules that reject "hits" (the result lags the real frame by k samples):

      L1  Within racket reach, higher than the feet: wrist keypoints / trajectory decide (as before).
      L2  Hit log (hit events): whenever a wrist moves fast near the ball (or L1 concludes a hit)
          record (frame, hitter's half, foot position). Then:
            - a candidate that bounces ON THE HALF OF THE PERSON WHO JUST HIT, within HIT_SIDE_EXCL_S after the hit
              -> rejected (a hit almost always goes to the opponent's side)
    Pixel thresholds are expressed at width 1920 and scale automatically with the video resolution.
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
        self.marks = []                                  # [(frame_idx, x, y)] mini-map coordinates
        self.swing = SwingDetector(fps)

        self.shown = {}            # half -> frame_idx of the most recent HIT report (for display)
        self.hits = []             # [[first frame, hitter's half, feet (mini-map) or None, last frame]]

    def reset(self):
        """Scene change: drop the old trajectory, bounce points and hit log."""
        self.samples.clear()
        self.marks.clear()
        self.swing.reset()
        self.hits.clear()
        self.shown.clear()

    # ------------------------------------------------------------------ hit log

    @staticmethod
    def _project_feet(H, players):
        feet = {}

        for p in players:
            pr = project_point(H, p["point"])

            if pr is not None:
                feet[p["half"]] = pr

        return feet

    def _register_hit(self, frame, half, feet):
        """Record a hit (from the wrist, the trajectory or inference). A signal overlapping an older hit is merged into it."""

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
            print(f"[Hit] frame {frame}: {half} hits the ball")

    def recent_hit_halves(self):
        """Set of halves currently within the HIT display time (from the wrist OR the ball trajectory)."""

        self.shown = {h: f for h, f in self.shown.items() if self.frame_idx - f <= self.show_frames}

        return set(self.shown)

    # ---- hit detection from the trajectory ----

    @staticmethod
    def _vel(frames, pts, a, b):
        """Velocity (px/frame) of samples a..b: slope of a linear regression over the frames."""
        return np.array([np.polyfit(frames[a:b + 1], pts[a:b + 1, k], 1)[0] for k in (0, 1)])

    @staticmethod
    def _player_of(players, half):
        """(cx, cy, h, y2) of the player in `half`, or None."""
        for p in players:
            if p["half"] == half:
                x1, y1, x2, y2 = p["box"]
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1), float(y2)
        return None

    @staticmethod
    def _reach_half(pt, players):
        """Half of the nearest player whose racket reach contains the ball (the ball need not be above the feet)."""

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
        Hit from the trajectory: ball within racket reach, direction reverses (>= HIT_TRAJ_TURN_DEG),
        approaches then leaves the player, and does not lose speed (a bounce loses speed). If the wrist
        has already confirmed "slow, no swing" (swing_state False), skip.
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
            return   # has not reversed enough

        players_b = players_seq[c]
        half = self._reach_half(pb, players_b)

        if half is None:
            return

        cx, cy, h, y2 = self._player_of(players_b, half)

        on_ground = pb[1] >= y2 - BALL_AIR_MIN_RAISE * h
        need = HIT_TRAJ_GROUND_RATIO if on_ground else HIT_TRAJ_SPEED_RATIO

        if s_out / s_in < need:
            return   # lost speed: looks more like a bounce than a hit

        # Approaches then leaves that same player
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
            print(f"[Hit] frame {fb}: trajectory reversal ({half}), speed after/before = {s_out / s_in:.2f}")

        self._register_hit(fb, half, feet_b)

    def _hit_context_reject(self, fb, half_b):
        """Returns the rejection reason (str) if the bounce candidate is actually a consequence of a hit, otherwise None."""

        # L2: according to the hit log (rejects only by court half; the "near the feet" rule was removed because it overlaps with this one)
        for first, half, _, _ in self.hits:
            dt = fb - first

            if dt < -self.hit_after_tol:
                continue   # the hit happens AFTER this bounce (bounce first, then hit): unrelated

            if dt <= self.hit_side_excl and half == half_b:
                return f"bounce on {half}'s half right after that person's hit"

        return None

    # ------------------------------------------------------------------ update

    def update(self, ball_point, players, H):
        """Call every frame. Returns (x, y) of the newly detected bounce point (mini-map coordinates) or None."""

        self.frame_idx += 1

        self.swing.update(self.frame_idx, players)   # record wrists every frame, before the early returns

        if ball_point is None or H is None:
            return None

        proj = project_point(H, ball_point)

        if proj is None:
            return None

        feet = self._project_feet(H, players)

        # Wrist swinging fast near the ball at the current frame -> record a hit (independent of the trajectory)
        hitter = self.swing.swing_hitter(self.frame_idx, ball_point)

        if hitter is not None:
            self._register_hit(self.frame_idx, hitter, feet)

        if self.samples and self.frame_idx - self.samples[-1][0] > self.max_gap:
            self.samples.clear()   # ball lost for too long: the old trajectory is no longer continuous

        self.samples.append(
            (self.frame_idx, np.array(ball_point, dtype=np.float32), proj, players, feet))

        if len(self.samples) < self.samples.maxlen:
            return None

        c = self.half
        frames = [s[0] for s in self.samples]
        pts = np.array([s[1] for s in self.samples])
        ys = pts[:, 1]

        fb, pb, proj_b, players_b, feet_b = self.samples[c]

        # Hit from the trajectory: runs every frame, independent of whether there is a bounce candidate
        self._trajectory_hit(fb, pb, frames, pts, [s[3] for s in self.samples], feet_b)

        min_drop = BOUNCE_MIN_DROP_PX * self.scale

        # Path 1: the middle sample is the trajectory bottom (largest image y) of the whole window
        is_bottom = (int(np.argmax(ys)) == c
                     and ys[c] - ys[0] >= min_drop and ys[c] - ys[-1] >= min_drop)

        # Path 2: vy drops abruptly even though image y does not reverse
        if not is_bottom and not self._is_impulse(frames, ys):
            return None

        if fb - self._last_bounce < self.cooldown:
            return None

        v_in = (pb - pts[0]) / (fb - frames[0])
        v_out = (pts[-1] - pb) / (frames[-1] - fb)

        # L1: ball within racket reach AND higher than the feet
        if self._in_racket_reach(pb, players_b):
            swing = self.swing.swing_state(fb, pb)
            hit_like = self._hit_like(v_in, v_out)

            if DEBUG_MODE:
                print(f"[Bounce] frame {fb}: within racket reach, swing={swing}, hit_like={hit_like}")

            if swing is True or (swing is None and hit_like):
                half_hit = self._reach_half(pb, players_b)

                if half_hit is not None:
                    self._register_hit(fb, half_hit, feet_b)

                return self._reject("hit within racket reach", fb)

        x, y = proj_b
        half_b = _half_of(y)

        # L2: hit context
        reason = self._hit_context_reject(fb, half_b)

        if reason is not None:
            return self._reject(reason, fb)

        if not (-BOUNCE_MAX_OUTSIDE_PX <= x <= MINI_COURT_W + BOUNCE_MAX_OUTSIDE_PX
                and -BOUNCE_MAX_OUTSIDE_PX <= y <= MINI_COURT_H + BOUNCE_MAX_OUTSIDE_PX):
            return self._reject("landed too far from the court", fb)

        self._last_bounce = fb
        self.marks.append((fb, float(x), float(y)))

        if DEBUG_MODE:
            print(f"[Bounce] frame {fb}: mini-map ({x:.0f}, {y:.0f}) half {half_b}")

        return float(x), float(y)

    # ------------------------------------------------------------------ utilities

    @staticmethod
    def _slope(frames, ys, a, b):
        return float(np.polyfit(frames[a:b + 1], ys[a:b + 1], 1)[0])

    def _is_impulse(self, frames, ys):
        """The ball is moving down and then vy drops abruptly (even if image y does not reverse)."""
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

        # Bounce = vy is jerked upward (d very negative). Do NOT require vin > 0: a ball bouncing obliquely away
        # from the camera is moving up in the image (vin <= 0) before the bounce, so image y has no "bottom".
        if d > -BOUNCE_MIN_DVY_PX * self.scale:
            return False

        if DEBUG_MODE:
            print(f"[Bounce] vy impulse: vin={vin:.1f} d={d:.1f}")

        return d <= min(v[0] for v in vals)

    def _reject(self, reason, frame):
        if DEBUG_MODE:
            print(f"[Bounce] frame {frame}: rejected ({reason})")
        return None

    def _hit_like(self, v_in, v_out):
        """A hit: the ball clearly speeds up or reverses horizontally."""

        speed_in = float(np.hypot(*v_in))
        speed_out = float(np.hypot(*v_out))

        if speed_in > 1e-6 and speed_out / speed_in > BOUNCE_HIT_SPEED_GAIN:
            return True

        vx_min = 2.0 * self.scale
        return v_in[0] * v_out[0] < 0 and abs(v_in[0]) > vx_min and abs(v_out[0]) > vx_min

    def recent_swings(self):
        """List of (half, wrist coordinates, box) for swings still within the display time."""
        return self.swing.recent_swings(self.frame_idx)

    def recent_marks(self):
        """List of (x, y) bounce points still within the display time (mini-map coordinates)."""

        self.marks = [m for m in self.marks if self.frame_idx - m[0] <= self.mark_frames]

        return [(x, y) for _, x, y in self.marks]

    @staticmethod
    def _in_racket_reach(pt, players):
        """True if the ball is within a player's racket reach and higher than the feet by more than BALL_AIR_MIN_RAISE."""

        bx, by = float(pt[0]), float(pt[1])

        for p in players:
            x1, y1, x2, y2 = p["box"]
            h = max(1.0, y2 - y1)

            in_reach = (x1 - HIT_REACH_X * h <= bx <= x2 + HIT_REACH_X * h
                        and y1 - HIT_REACH_UP * h <= by <= y2)

            if in_reach and by <= y2 - BALL_AIR_MIN_RAISE * h:
                return True

        return False
