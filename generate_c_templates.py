import cv2
import os
from pathlib import Path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    project_root = Path(__file__).resolve().parent
    frame_path = project_root / "frames_backup/frame_003880.jpg"
    template_index = 3

    if not frame_path.exists():
        raise FileNotFoundError(f"Frame bulunamadı: {frame_path}")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Görüntü okunamadı: {frame_path}")

    clone = img.copy()
    cv2.imshow("C ROI sec - Enter ile onayla, ESC ile iptal", clone)

    roi = cv2.selectROI("C ROI sec - Enter ile onayla, ESC ile iptal",
                        clone, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        return

    c_patch = img[y:y + h, x:x + w]
    c_patch = cv2.rotate(c_patch, cv2.ROTATE_180)

    templates_root = project_root / "templates" / "c_signs"
    right_dir = templates_root / "right"
    left_dir = templates_root / "left"
    up_dir = templates_root / "up"
    down_dir = templates_root / "down"

    for d in [right_dir, left_dir, up_dir, down_dir]:
        ensure_dir(d)

    right_path = right_dir / f"c_right_{template_index}.png"
    cv2.imwrite(str(right_path), c_patch)

    up_patch = cv2.rotate(c_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    up_path = up_dir / f"c_up_{template_index}.png"
    cv2.imwrite(str(up_path), up_patch)

    left_patch = cv2.rotate(c_patch, cv2.ROTATE_180)
    left_path = left_dir / f"c_left_{template_index}.png"
    cv2.imwrite(str(left_path), left_patch)

    down_patch = cv2.rotate(c_patch, cv2.ROTATE_90_CLOCKWISE)
    down_path = down_dir / f"c_down_{template_index}.png"
    cv2.imwrite(str(down_path), down_patch)

    print("[BILGI] C template'leri kaydedildi:")
    print(f" - RIGHT: {right_path}")
    print(f" - UP   : {up_path}")
    print(f" - LEFT : {left_path}")
    print(f" - DOWN : {down_path}")


if __name__ == "__main__":
    main()
