from __future__ import annotations

import cv2
import argparse
import os
from pathlib import Path


def extract_frames(
        video_path: str,
        output_dir: str,
        step: int = 1,
        max_duration_sec: float | None = None
):
    """
    Verilen videoyu frame'lere ayırır ve her step'te bir frame'i output_dir'e kaydeder.

    :param video_path: Girdi video dosyası yolu
    :param output_dir: Çıkacak frame görüntülerinin kaydedileceği klasör
    :param step: Kaç frame'de bir kayıt yapılacağı (1 = her frame)
    :param max_duration_sec: Maksimum süre (saniye). None ise tüm video.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    # Çıktı klasörünü oluştur
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {video_path.name}")
    print(f"[INFO] FPS: {fps:.2f}")
    print(f"[INFO] Toplam frame: {total_frames}")

    # 7 dakika kontrolü (max_duration_sec parametresi ile)
    if max_duration_sec is not None:
        max_frames = int(fps * max_duration_sec)
        print(f"[INFO] Maksimum süre: {max_duration_sec} sn → {max_frames} frame")
    else:
        max_frames = total_frames

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Video bitti
            break

        # Süre sınırı varsa
        if frame_idx >= max_frames:
            print("[INFO] Maksimum süreye ulaşıldı, işleme son veriliyor.")
            break

        # Her step'te bir frame kaydet
        if frame_idx % step == 0:
            # Zaman bilgisi (saniye)
            timestamp_sec = frame_idx / fps if fps > 0 else 0.0

            # Dosya ismi: frame_index ve zaman ile
            filename = f"frame_{frame_idx:06d}_t{timestamp_sec:07.2f}.png"
            save_path = output_dir / filename

            cv2.imwrite(str(save_path), frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"[INFO] {saved_count} frame kaydedildi...")

        frame_idx += 1

    cap.release()
    print(f"[DONE] Toplam kaydedilen frame sayısı: {saved_count}")
    print(f"[DONE] Çıktı klasörü: {output_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Videoyu frame'lere ayıran basit görüntü işleme aracı."
    )
    parser.add_argument("video", help="Girdi video dosyasının yolu")
    parser.add_argument(
        "-o", "--output",
        default="frames",
        help="Frame çıktılarının kaydedileceği klasör (varsayılan: frames)"
    )
    parser.add_argument(
        "-s", "--step",
        type=int,
        default=1,
        help="Kaç frame'de bir kayıt yapılacağı (varsayılan: 1 = her frame)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=420.0,  # 7 dakika = 420 saniye
        help="Maksimum video süresi (saniye cinsinden, varsayılan: 420 sn = 7 dk)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        step=args.step,
        max_duration_sec=args.max_duration
    )
