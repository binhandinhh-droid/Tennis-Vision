
import cv2
import config
from court_detector import project_point


def classify_player(foot_point, H, mini_w, mini_h):
    """
    Chiếu foot_point qua homography H sang không gian sân chuẩn hoá
    (0..mini_w, 0..mini_h) rồi phân loại + lọc theo margin thật (mét).

    Returns:
        (label, priority) hoặc None:
        label    = "Player 1" (nửa dưới) hoặc "Player 2" (nửa trên)
        priority = 0 nếu nằm hẳn trong biên sân thật, càng âm càng
                   nằm xa ngoài biên thật (dù vẫn trong margin cho phép)
        None     = ngoài vùng sân hợp lệ -> bỏ qua (khán giả, ballkid...)
    """

    if H is None:
        return None

    proj = project_point(H, foot_point)

    if proj is None:
        return None

    px, py = proj

    margin_x = mini_w * (config.PLAYER_MARGIN_SIDE_M / config.COURT_WIDTH_M)
    margin_y = mini_h * (config.PLAYER_MARGIN_DEPTH_M / config.COURT_LENGTH_M)

    if px < -margin_x or px > mini_w + margin_x:
        return None

    if py < -margin_y or py > mini_h + margin_y:
        return None

    label = "Player 2" if py < mini_h / 2.0 else "Player 1"

    # Khoảng cách (âm) ra ngoài biên sân THẬT (chưa tính margin) -
    # 0 nghĩa là điểm nằm hẳn trong sân.
    outside_x = max(0.0, -px, px - mini_w)
    outside_y = max(0.0, -py, py - mini_h)

    priority = -(outside_x + outside_y)

    return label, priority


def select_players(xyxy, ids, H, mini_w, mini_h):
    """
    Từ danh sách bbox người detect được, chọn đúng 1 candidate tốt
    nhất mỗi nửa sân (Player 1 / Player 2).

    Args:
        xyxy: (N, 4) array bbox người
        ids:  (N,) array track id hoặc None
        H:    homography hiện tại

    Returns:
        dict {"Player 1": candidate|None, "Player 2": candidate|None}
        candidate = {"point", "label", "priority", "area", "box", "id"}
    """

    best_candidate = {"Player 1": None, "Player 2": None}

    for i, box in enumerate(xyxy):

        x1, y1, x2, y2 = map(int, box)

        foot_point = ((x1 + x2) / 2.0, y2)

        result = classify_player(foot_point, H, mini_w, mini_h)

        if result is None:
            continue

        label, priority = result

        area = max(0, x2 - x1) * max(0, y2 - y1)
        track_id = int(ids[i]) if ids is not None else None

        candidate = {
            "point": foot_point,
            "label": label,
            "priority": priority,
            "area": area,
            "box": (x1, y1, x2, y2),
            "id": track_id,
        }

        current_best = best_candidate[label]

        if current_best is None or (candidate["priority"], candidate["area"]) > \
                (current_best["priority"], current_best["area"]):
            best_candidate[label] = candidate

    return best_candidate


def draw_players(annotated_frame, best_candidate, debug=config.DEBUG_MODE):
    """Vẽ box + nhãn cho đúng 1 candidate tốt nhất mỗi nửa sân.

    Returns:
        list các điểm chân + nhãn, dùng để vẽ lên mini-map.
    """

    mini_map_players = []

    for label, candidate in best_candidate.items():

        if candidate is None:
            continue

        x1, y1, x2, y2 = candidate["box"]

        mini_map_players.append({
            "point": candidate["point"],
            "label": label,
        })

        color = (0, 255, 0) if label == "Player 1" else (255, 128, 0)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label_text = label

        if candidate["id"] is not None and debug:
            label_text += f" (id {candidate['id']})"

        cv2.putText(
            annotated_frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    return mini_map_players


def track_players(frame, annotated_frame, person_model, cached_H, mini_w, mini_h,
                   debug=config.DEBUG_MODE):
    """
    Chạy YOLO tracking (ByteTrack) trên frame, lọc + gán đúng 1 player
    mỗi nửa sân, rồi vẽ box/nhãn lên annotated_frame (in-place).

    Returns:
        mini_map_players (list) - dùng để vẽ lên mini-map.
    """

    if cached_H is None:
        return []

    person_results = person_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False,
    )

    if not len(person_results):
        return []

    boxes = person_results[0].boxes

    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()

    ids = None
    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)

    best_candidate = select_players(xyxy, ids, cached_H, mini_w, mini_h)

    return draw_players(annotated_frame, best_candidate, debug)
