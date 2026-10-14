"""Tennis court mini-map (top-down) drawn in the top-right corner of the frame."""

import cv2
import numpy as np

from config import (
    MINI_COURT_W, MINI_COURT_H, MINI_PADDING, MINI_MARGIN_TOP, MINI_MARGIN_RIGHT,
    SINGLES_INSET_RATIO, SERVICE_LINE_RATIO_NEAR, SERVICE_LINE_RATIO_FAR, PLAYER_COLORS,
)
from geometry import project_point

_MINI_CANVAS_W = MINI_COURT_W + MINI_PADDING * 2
_MINI_CANVAS_H = MINI_COURT_H + MINI_PADDING * 2


def _build_mini_court_base():
    """Static mini-map background (court outline, singles sidelines, service lines, center line, net) - drawn once."""

    canvas = np.full((_MINI_CANVAS_H, _MINI_CANVAS_W, 3), (25, 25, 25), dtype=np.uint8)

    w, h, pad = MINI_COURT_W, MINI_COURT_H, MINI_PADDING
    white = (255, 255, 255)

    cv2.rectangle(canvas, (pad, pad), (pad + w, pad + h), white, 2)

    inset = int(w * SINGLES_INSET_RATIO)

    cv2.line(canvas, (pad + inset, pad), (pad + inset, pad + h), white, 1)
    cv2.line(canvas, (pad + w - inset, pad), (pad + w - inset, pad + h), white, 1)

    near_y = int(h * SERVICE_LINE_RATIO_NEAR)
    far_y = int(h * SERVICE_LINE_RATIO_FAR)

    cv2.line(canvas, (pad + inset, pad + near_y), (pad + w - inset, pad + near_y), white, 1)
    cv2.line(canvas, (pad + inset, pad + far_y), (pad + w - inset, pad + far_y), white, 1)

    center_x = pad + w // 2
    cv2.line(canvas, (center_x, pad + near_y), (center_x, pad + far_y), white, 1)

    net_y = pad + h // 2
    cv2.line(canvas, (pad, net_y), (pad + w, net_y), (0, 255, 255), 2)

    return canvas


_MINI_COURT_BASE = _build_mini_court_base()


def _blit_canvas(frame, canvas):
    """Paste the mini-map canvas onto the top-right corner of the frame."""

    x_off = frame.shape[1] - _MINI_CANVAS_W - MINI_MARGIN_RIGHT
    y_off = MINI_MARGIN_TOP

    if x_off < 0 or y_off < 0:
        return

    roi = frame[y_off:y_off + _MINI_CANVAS_H, x_off:x_off + _MINI_CANVAS_W]
    frame[y_off:y_off + _MINI_CANVAS_H, x_off:x_off + _MINI_CANVAS_W] = \
        cv2.addWeighted(roi, 0.25, canvas, 0.75, 0)

    cv2.rectangle(
        frame, (x_off, y_off), (x_off + _MINI_CANVAS_W, y_off + _MINI_CANVAS_H),
        (255, 255, 255), 1,
    )


def draw_mini_court(frame, H, players, ball_point, marks=None, swing_halves=None):
    """
    Draw the tennis court mini-map (top-down) in the top-right corner of the frame.

    Args:
        H:          homography of the current frame
        players:    list of dicts {"point": (x, y), "label": "Player 1"/"Player 2", ...}
        ball_point: (x, y) ball center on the original frame, or None
        swing_halves: set of halves ("Player 1"/"Player 2") that are swinging: draw a red ring + "HIT" text around the player dot
        marks:      list of (x, y) recent bounce points (mini-map coordinates), drawn as X marks; or None
    """

    canvas = _MINI_COURT_BASE.copy()
    pad = MINI_PADDING

    for p in players:

        proj = project_point(H, p["point"])

        if proj is None:
            continue

        px = int(np.clip(proj[0], 0, MINI_COURT_W))
        py = int(np.clip(proj[1], 0, MINI_COURT_H))

        cv2.circle(canvas, (pad + px, pad + py), 6, PLAYER_COLORS[p["label"]], -1)
        cv2.circle(canvas, (pad + px, pad + py), 6, (0, 0, 0), 1)

        if swing_halves and p["half"] in swing_halves:
            red = (0, 0, 255)
            cv2.circle(canvas, (pad + px, pad + py), 13, red, 3)
            tx = pad + px + 17 if px < MINI_COURT_W / 2 else pad + px - 52
            cv2.putText(canvas, "HIT", (tx, pad + py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)

    for mx, my in (marks or []):
        cx = pad + int(np.clip(mx, -10, MINI_COURT_W + 10))
        cy = pad + int(np.clip(my, -10, MINI_COURT_H + 10))
        cv2.line(canvas, (cx - 4, cy - 4), (cx + 4, cy + 4), (0, 255, 255), 2)
        cv2.line(canvas, (cx - 4, cy + 4), (cx + 4, cy - 4), (0, 255, 255), 2)

    if ball_point is not None:

        proj = project_point(H, ball_point)

        if proj is not None:
            bx, by = proj

            # Only draw if the ball is near the court (allow it to fly slightly outside the lines)
            if -30 <= bx <= MINI_COURT_W + 30 and -30 <= by <= MINI_COURT_H + 30:
                bx = int(np.clip(bx, 0, MINI_COURT_W))
                by = int(np.clip(by, 0, MINI_COURT_H))

                cv2.circle(canvas, (pad + bx, pad + by), 4, (0, 0, 255), -1)

    _blit_canvas(frame, canvas)
