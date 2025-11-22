import cv2
import os
import sys

MAX_DURATION_SEC = 7 * 60

def extract_frames(video_path,output_dir="frames_backup"):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration_sec = total_frames / fps if fps > 0 else 0

    if duration_sec > MAX_DURATION_SEC:
        cap.release()
        raise ValueError(f"Video 7 dakikadan uzun! Süre: {duration_sec / 60:.2f} dk")

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)

        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Toplam {frame_idx} frame kaydedildi. Çıkış klasörü: {output_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Kullanım: python script.py video_dosyasi.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    extract_frames(video_path)
