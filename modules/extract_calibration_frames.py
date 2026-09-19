import os
import cv2
import config


def extract_calibration_frames(
    input_video=config.INPUT_VIDEO,
    output_dir=config.CALIBRATION_DATA_DIR,
    num_frames=config.CALIBRATION_NUM_FRAMES,
):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Không đọc được số lượng frame của video.")

    num_frames = min(num_frames, total_frames)

    # Chỉ số frame rải đều trên toàn bộ video, để tập calibration đại
    # diện được nhiều tình huống (góc quay, ánh sáng, vị trí bóng/người
    # khác nhau) thay vì chỉ lấy đoạn đầu video.
    frame_indices = sorted(set(
        int(i * total_frames / num_frames) for i in range(num_frames)
    ))

    saved = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        img_path = os.path.join(images_dir, f"calib_{idx:06d}.jpg")
        cv2.imwrite(img_path, frame)

        # Tạo sẵn file label rỗng - dataset loader của ultralytics cần
        # thư mục labels tồn tại song song với images để không lỗi khi
        # đọc dataset. Calibration INT8 chỉ dùng ảnh, không cần nhãn
        # thật, nên để trống là đủ.
        label_path = os.path.join(labels_dir, f"calib_{idx:06d}.txt")
        open(label_path, "w").close()

        saved += 1

    cap.release()

    print(f"Đã trích {saved}/{num_frames} frame calibration vào: {images_dir}")

    return images_dir


if __name__ == "__main__":
    extract_calibration_frames()
