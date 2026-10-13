"""Track người chơi: YOLO-pose trên vùng quanh sân + PlayerMemory giữ đúng 1 player mỗi nửa sân.

Không dùng ByteTrack của ultralytics nữa (id đổi / mất khi bị che làm player "biến mất").
Mỗi frame chỉ lấy detection thô, PlayerMemory tự nối người giữa các frame theo vị trí:
  - hụt detection -> giữ vị trí cũ tối đa PLAYER_COAST_FRAMES frame (coast)
  - người khác chỉ thay thế khi tốt hơn hẳn liên tục PLAYER_SWITCH_FRAMES frame
"""

import cv2

from config import (
    DEBUG_MODE, PLAYER_COLORS, MINI_COURT_W, MINI_COURT_H,
    PERSON_CROP_MARGIN_X, PERSON_CROP_MARGIN_Y, _PLAYER_MARGIN_X, _PLAYER_MARGIN_Y,
    PERSON_CONF, PERSON_IMGSZ,
    PLAYER_COAST_FRAMES, PLAYER_MATCH_DIST_H, PLAYER_MATCH_GROWTH,
    PLAYER_SWITCH_FRAMES, PLAYER_SWITCH_PRIORITY_GAP,
)
from geometry import project_point

HALVES = ("Player 1", "Player 2")


def classify_player(foot_point, H):
    """
    Chiếu foot_point qua H rồi phân loại + lọc theo margin thật (mét).

    Returns:
        (label, priority) hoặc None nếu nằm ngoài vùng hợp lệ.
        label    = "Player 1" (nửa dưới) / "Player 2" (nửa trên)
        priority = 0 nếu nằm trong biên sân, càng âm càng xa biên.
    """

    proj = project_point(H, foot_point)

    if proj is None:
        return None

    px, py = proj

    if px < -_PLAYER_MARGIN_X or px > MINI_COURT_W + _PLAYER_MARGIN_X:
        return None

    if py < -_PLAYER_MARGIN_Y or py > MINI_COURT_H + _PLAYER_MARGIN_Y:
        return None

    label = "Player 2" if py < MINI_COURT_H / 2.0 else "Player 1"

    outside_x = max(0.0, -px, px - MINI_COURT_W)
    outside_y = max(0.0, -py, py - MINI_COURT_H)

    return label, -(outside_x + outside_y)


def build_candidates(xyxy, confs, kpts, H):
    """Gom detection thành {half: [candidate,...]} (đã lọc theo vùng hợp lệ, xếp tốt nhất trước)."""

    cands = {h: [] for h in HALVES}

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        foot_point = ((x1 + x2) / 2.0, y2)

        result = classify_player(foot_point, H)

        if result is None:
            continue

        label, priority = result

        cands[label].append({
            "label": label,
            "half": label,   # nửa sân (Player 1 = gần camera)
            "point": foot_point,
            "box": (x1, y1, x2, y2),
            "id": None,
            "conf": float(confs[i]),
            "rank": (priority, max(0, x2 - x1) * max(0, y2 - y1)),
            "kpts": kpts[i] if kpts is not None else None,   # (17, 3) toạ độ frame gốc
            "coasted": False,
        })

    for h in HALVES:
        cands[h].sort(key=lambda c: c["rank"], reverse=True)

    return cands


class PlayerMemory:
    """Giữ ổn định đúng 1 player mỗi nửa sân qua các frame."""

    def __init__(self):
        self.state = {}   # half -> {"player", "miss", "challenger"}

    def reset(self):
        self.state.clear()

    def update(self, cands):
        out = []

        for half in HALVES:
            p = self._update_half(half, cands.get(half, []))

            if p is not None:
                out.append(p)

        return out

    @staticmethod
    def _match(prev, miss, cands):
        """Candidate gần prev nhất, trong bán kính tính theo chiều cao box (nở dần khi hụt lâu)."""

        px1, py1, px2, py2 = prev["box"]
        pc = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
        radius = (PLAYER_MATCH_DIST_H + PLAYER_MATCH_GROWTH * miss) * max(1.0, py2 - py1)

        best, best_d = None, radius

        for c in cands:
            x1, y1, x2, y2 = c["box"]
            d = ((((x1 + x2) / 2.0) - pc[0]) ** 2 + (((y1 + y2) / 2.0) - pc[1]) ** 2) ** 0.5

            if d <= best_d:
                best, best_d = c, d

        return best

    def _update_half(self, half, cands):
        st = self.state.get(half)
        best = cands[0] if cands else None

        if st is None:
            if best is None:
                return None

            self.state[half] = {"player": best, "miss": 0, "challenger": 0}
            return best

        matched = self._match(st["player"], st["miss"], cands)

        if matched is not None:
            st["miss"] = 0

            # Người khác tốt hơn hẳn (gần sân hơn) liên tục N frame mới thay
            if (best is not None and best is not matched
                    and best["rank"][0] - matched["rank"][0] >= PLAYER_SWITCH_PRIORITY_GAP):
                st["challenger"] += 1
            else:
                st["challenger"] = 0

            if st["challenger"] >= PLAYER_SWITCH_FRAMES:
                matched = best
                st["challenger"] = 0

            st["player"] = matched
            return matched

        # Không khớp: coast vị trí cũ
        st["miss"] += 1
        st["challenger"] = 0

        if st["miss"] > PLAYER_COAST_FRAMES:
            if best is None:
                del self.state[half]
                return None

            self.state[half] = {"player": best, "miss": 0, "challenger": 0}
            return best

        coast = dict(st["player"])
        coast["kpts"] = None       # không có keypoint mới
        coast["coasted"] = True
        st["player"] = coast
        return coast


def _is_fixed_shape(model):
    name = str(getattr(model, "model_name", "") or getattr(model, "ckpt_path", "") or "")
    return name.lower().endswith(".engine")


def _crop_around_court(frame, court_box):
    """
    Crop vùng quanh sân (+ margin để vẫn thấy người đứng sau baseline) đưa vào YOLO
    thay cho cả frame. Trả về (source, x_off, y_off); không crop nếu vùng crop
    quá nhỏ hoặc không nhỏ hơn hẳn cả frame.
    """

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = court_box
    bw, bh = x2 - x1, y2 - y1

    cx1 = int(max(0, x1 - PERSON_CROP_MARGIN_X * bw))
    cy1 = int(max(0, y1 - PERSON_CROP_MARGIN_Y * bh))
    cx2 = int(min(w, x2 + PERSON_CROP_MARGIN_X * bw))
    cy2 = int(min(h, y2 + PERSON_CROP_MARGIN_Y * bh))

    cw, ch = cx2 - cx1, cy2 - cy1

    if cw < 64 or ch < 64 or cw * ch >= 0.9 * h * w:
        return frame, 0, 0

    return frame[cy1:cy2, cx1:cx2], cx1, cy1


class PlayerTracker:
    """YOLO-pose (predict, không ByteTrack) + PlayerMemory."""

    def __init__(self, person_model):
        self.model = person_model
        self._dynamic_imgsz = not _is_fixed_shape(person_model)
        self.memory = PlayerMemory()

    def reset(self):
        """Chuyển cảnh: bỏ người đang theo dõi."""
        self.memory.reset()

    def update(self, frame, H, court_box):
        """Returns list {"point", "label", "half", "box", "kpts", "coasted", ...} (đúng ≤1 người / nửa sân)."""

        source, x_off, y_off = _crop_around_court(frame, court_box)

        kwargs = {"imgsz": PERSON_IMGSZ} if self._dynamic_imgsz else {}

        res = self.model.predict(
            source, conf=PERSON_CONF, classes=[0], verbose=False, **kwargs,
        )[0]
        boxes = res.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy[:, [0, 2]] += x_off
        xyxy[:, [1, 3]] += y_off

        # Keypoint (chỉ có khi dùng model pose); đưa về toạ độ frame gốc như box
        kpts = None

        if res.keypoints is not None and res.keypoints.data.numel() > 0 and len(boxes) > 0:
            kpts = res.keypoints.data.cpu().numpy().copy()   # (N, 17, 3)

            if kpts.shape[1] >= 11:
                kpts[:, :, 0] += x_off
                kpts[:, :, 1] += y_off
            else:
                kpts = None

        return self.memory.update(build_candidates(xyxy, confs, kpts, H))


def _draw_player(img, player):
    x1, y1, x2, y2 = player["box"]
    color = PLAYER_COLORS[player["label"]]

    text = player["label"]

    if DEBUG_MODE:
        text += f" {player.get('conf', 0):.2f}" + (" (coast)" if player.get("coasted") else "")

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if player.get("coasted") else 2)
    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def draw_players(img, players):
    """Vẽ box + nhãn từng player ."""

    for p in players:
        _draw_player(img, p)
