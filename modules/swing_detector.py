"""Racket swing detection from the players' wrist keypoints (YOLO-pose)."""

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
    Evidence of a racket swing from wrist motion RELATIVE TO THE TORSO (shoulder + hip center):
      - camera pan / zoom, or a person running / jumping, moves the torso and wrists together, so it does not count as a swing;
      - speed is measured in person heights / second (independent of camera distance and fps),
        over SWING_SPEED_SPAN frames so a few px of keypoint jitter is not counted;
      - unrealistic speeds (> SWING_MAX_SPEED_H_PER_S: keypoint jumps, left/right flips) are discarded;
      - the wrist must be fast for >= SWING_MIN_FRAMES consecutive frames; 1-frame noise is not enough.

    `swing_state` (used to filter bounces) returns 3 states:
        True   wrist swinging fast AND close to the ball  -> definitely a hit
        False  wrist near the ball, clearly visible, moving slowly -> definitely NOT a hit
        None   not enough information (wrist / torso not visible, or speed in between)

    `recent_swings` (used for display) reports a swing right at the current frame, without needing the ball.
    """

    def __init__(self, fps):
        self.fps = fps if fps and fps > 0 else 30.0
        self.tol = max(1, int(round(SWING_TOL_S * self.fps)))
        self.hist = {}      # half -> deque[(frame_idx, wrists_abs (2,2), wrists_rel (2,2), h)]
        self.streak = {}    # half -> (last frame_idx, number of consecutive fast frames)

        # Swing display: half -> (frame_idx, wrist coordinates, player box) of the most recent swing
        self.show_frames = max(1, int(round(SWING_SHOW_S * self.fps)))
        self.last_swing = {}

    def reset(self):
        self.hist.clear()
        self.streak.clear()
        self.last_swing.clear()

    # ------------------------------------------------------------------ recording

    @staticmethod
    def _extract(kp):
        """(wrists_abs (2,2), wrists_rel (2,2)). rel = wrist - torso center; NaN if a keypoint is missing."""

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
        """Call every frame (even when the ball is not seen) to keep the wrist history continuous."""

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
        Wrist speed relative to the torso at frame f (person heights / second), shape (2,).
        NaN if samples are missing / wrist or torso not visible / above SWING_MAX_SPEED_H_PER_S.
        """

        rel = by_frame[f][1]

        for span in range(SWING_SPEED_SPAN, 0, -1):   # if samples are missing, try a shorter span
            if f - span in by_frame:
                with np.errstate(invalid="ignore"):
                    speeds = np.linalg.norm(rel - by_frame[f - span][1], axis=1) / span / h * self.fps
                    speeds[speeds > SWING_MAX_SPEED_H_PER_S] = np.nan

                return speeds

        return np.full(2, np.nan, dtype=np.float32)

    def _mark_swing(self, player, dq, frame_idx, h):
        """Report a swing RIGHT at the current frame (do not wait for later frames, no ball needed)."""

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
            j = int(np.nanargmax(speeds))   # fastest wrist = the racket hand
            self.last_swing[half] = (frame_idx, by_frame[frame_idx][0][j].copy(), player["box"])

    # ------------------------------------------------------------------ queries

    def recent_swings(self, frame_idx):
        """List of (half, wrist coordinates, box) for swings still within the SWING_SHOW_S display time."""

        self.last_swing = {
            half: s for half, s in self.last_swing.items() if frame_idx - s[0] <= self.show_frames
        }

        return [(half, s[1], s[2]) for half, s in self.last_swing.items()]

    def _wrist_speeds(self, dq, frame_idx):
        """For each frame f around frame_idx: (f, wrists_abs, h, speeds (2,))."""

        by_frame = {f: (wr, rel) for f, wr, rel, _ in dq}
        heights = {f: h for f, _, _, h in dq}
        out = []

        for f in range(frame_idx - self.tol, frame_idx + self.tol + 1):
            if f in by_frame:
                out.append((f, by_frame[f][0], heights[f], self._speeds_at(by_frame, f, heights[f])))

        return out

    def _scan(self, frame_idx, ball_pt):
        """half -> (fast_near, seen_slow, seen_mid) around frame_idx. See swing_state."""

        ball = np.asarray(ball_pt, dtype=np.float32)
        out = {}

        for half, dq in self.hist.items():
            fast_near = 0   # number of frames with a wrist that is fast AND close to the ball
            seen_slow = 0   # number of wrist samples CLOSE to the ball, clearly visible and slow
            seen_mid = 0    # number of samples near the ball but with in-between speed

            for _, wr, h, speeds in self._wrist_speeds(dq, frame_idx):

                with np.errstate(invalid="ignore"):
                    dist = np.linalg.norm(wr - ball, axis=1) / h   # NaN -> every comparison is False

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
        """True / False / None, see the class docstring."""

        scan = self._scan(frame_idx, ball_pt)

        fast_near = sum(v[0] for v in scan.values())
        seen_slow = sum(v[1] for v in scan.values())
        seen_mid = sum(v[2] for v in scan.values())

        if fast_near >= min(SWING_MIN_FRAMES, 2 * self.tol + 1):
            return True

        # Must see slow motion in >= 2 samples and no in-between sample before daring to assert "no swing"
        if seen_slow >= 2 and seen_mid == 0:
            return False

        return None

    def swing_hitter(self, frame_idx, ball_pt):
        """Half of the person swinging near the ball (wrist fast, close to the ball) at frame_idx, or None."""

        need = min(SWING_MIN_FRAMES, 2 * self.tol + 1)
        best = None

        for half, (fast_near, _, _) in self._scan(frame_idx, ball_pt).items():
            if fast_near >= need and (best is None or fast_near > best[1]):
                best = (half, fast_near)

        return best[0] if best else None

    def swing_near_ball(self, frame_idx, ball_pt):
        """Backward compatible: True if there is definitely a swing near the ball."""
        return self.swing_state(frame_idx, ball_pt) is True
