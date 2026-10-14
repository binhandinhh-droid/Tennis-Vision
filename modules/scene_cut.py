"""Scene change detection (cut / dissolve)."""

from collections import deque

import cv2
import numpy as np

from config import (
    DEBUG_MODE, SCENE_CUT_HIST_CORR, SCENE_CUT_MAD, SCENE_CUT_WINDOW, SCENE_DISSOLVE_CORR,
    SCENE_LOST_S, SCENE_LOST_BALL_RATIO, SCENE_BANNER_S,
)


class SceneCutDetector:
    """
    Detects a scene change between two consecutive frames: the (downscaled) HSV histogram shifts strongly
    or the grayscale image changes strongly (hard cut).
    Plus one comparison against the frame SCENE_CUT_WINDOW frames back to catch gradual transitions (dissolve / wipe),
    since each pair of consecutive frames in a dissolve differs only slightly.
    Camera pan / zoom within the same scene changes slowly, so it is not counted.
    """

    def __init__(self):
        self.hists = deque(maxlen=SCENE_CUT_WINDOW + 1)
        self.prev_gray = None
        self.since_cut = 10 ** 9

    def update(self, frame, small=None):
        """Returns True if this frame is the first frame of a new scene.
        small: frame already downscaled to 160x90 (precomputed in the reader thread); computed here if None."""

        if small is None:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten().astype(np.float32)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        cut = False

        if self.hists:
            corr = cv2.compareHist(self.hists[-1], hist, cv2.HISTCMP_CORREL)
            mad = float(np.abs(gray - self.prev_gray).mean())
            cut = corr < SCENE_CUT_HIST_CORR or mad > SCENE_CUT_MAD

            # Gradual transition: compare with the frame SCENE_CUT_WINDOW frames back (and no cut was just reported)
            if (not cut and len(self.hists) == self.hists.maxlen
                    and self.since_cut > SCENE_CUT_WINDOW):
                corr_w = cv2.compareHist(self.hists[0], hist, cv2.HISTCMP_CORREL)
                cut = corr_w < SCENE_DISSOLVE_CORR

                if DEBUG_MODE and cut:
                    print(f"[Scene] gradual transition corr_window={corr_w:.2f}")

            if DEBUG_MODE and (cut or corr < 0.9):
                print(f"[Scene] corr={corr:.2f} mad={mad:.1f} cut={cut}")

        self.since_cut = 0 if cut else self.since_cut + 1

        self.hists.append(hist)
        self.prev_gray = gray

        return cut


class SceneMonitor(SceneCutDetector):
    """
    Scene change = strong histogram/image change (SceneCutDetector)  OR  loss of the court signal:
        no court (low-confidence court detection / lost track) for SCENE_LOST_S seconds in a row
        AND the ball is almost never seen in that period AND there are no players  -> "lost"
    When the court reappears after "lost" -> "back" (also a scene change).
    On every scene change the text "SCENE CHANGE" is shown on screen for SCENE_BANNER_S seconds.
    """

    def __init__(self, fps):
        super().__init__()

        fps = fps if fps and fps > 0 else 30.0
        self.lost_frames = max(3, int(round(SCENE_LOST_S * fps)))
        self.banner_frames = max(1, int(round(SCENE_BANNER_S * fps)))

        self.court_miss = 0
        self.ball_hist = deque(maxlen=self.lost_frames)
        self.in_lost = False
        self.banner_left = 0

    def _trigger(self):
        if self.banner_left == 0:   # a banner from the previous scene change is still showing: do not restart it
            self.banner_left = self.banner_frames

    def update(self, frame, small=None):
        """Visual scene change. Returns True if this frame is the first frame of a new scene."""

        cut = super().update(frame, small)

        if cut:
            self._trigger()
            self.court_miss = 0
            self.ball_hist.clear()

        return cut

    def update_signals(self, court_found, ball_found, players_found):
        """Call every frame after court / players / ball have run. Returns None | "lost" | "back"."""

        self.court_miss = 0 if court_found else self.court_miss + 1
        self.ball_hist.append(bool(ball_found))

        if not self.in_lost:
            if (self.court_miss >= self.lost_frames
                    and len(self.ball_hist) == self.ball_hist.maxlen
                    and sum(self.ball_hist) / len(self.ball_hist) <= SCENE_LOST_BALL_RATIO
                    and not players_found):
                self.in_lost = True
                self._trigger()

                if DEBUG_MODE:
                    print("[Scene] court lost + ball lost -> switched to a non-court scene")

                return "lost"

        elif court_found:
            self.in_lost = False
            self.ball_hist.clear()
            self._trigger()

            if DEBUG_MODE:
                print("[Scene] court seen again -> new scene")

            return "back"

        return None

    def draw(self, img):
        """Draw the SCENE CHANGE text at the top center of the frame for banner_frames frames."""

        if self.banner_left <= 0:
            return

        self.banner_left -= 1

        text = "SCENE CHANGE"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = (img.shape[1] - tw) // 2
        y = 80

        cv2.rectangle(img, (x - 16, y - th - 14), (x + tw + 16, y + 14), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, scale, (0, 165, 255), thick)
