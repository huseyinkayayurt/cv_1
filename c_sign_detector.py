import cv2
import numpy as np
from pathlib import Path
import math


class CSignDetector:
    """
    C işaretini (Sağ / Sol / Yukarı / Aşağı) template matching ile tespit eder,
    videoyu oynatır, yeni obje bulununca videoyu durdurur ve komut satırına yazdırır.
    """

    # Klasör adı -> Türkçe yön ismi
    DIR_LABELS = {
        "right": "Sağ",
        "left": "Sol",
        "up": "Yukarı",
        "down": "Aşağı",
    }

    def __init__(
        self,
        templates_root: str,
        match_threshold: float = 0.6,
        same_object_max_distance: float = 60.0,
        pause_key: str = " "
    ):
        self.templates_root = Path(templates_root)
        self.match_threshold = match_threshold
        self.same_object_max_distance = same_object_max_distance
        self.pause_key_code = ord(pause_key) if len(pause_key) == 1 else 32

        # { "Sağ": [ (template_img, (w, h)), ... ], ... }
        self.templates = self._load_templates()

        # Daha önce tespit edilmiş objeler: [ (direction, (cx, cy)) , ... ]
        self.detected_objects = []

    # -----------------------------
    # Template yükleme
    # -----------------------------
    def _load_templates(self):
        if not self.templates_root.exists():
            raise FileNotFoundError(f"Template klasörü bulunamadı: {self.templates_root}")

        templates = {label: [] for label in self.DIR_LABELS.values()}

        for dir_name, direction_label in self.DIR_LABELS.items():
            dir_path = self.templates_root / dir_name
            if not dir_path.is_dir():
                # Bu yön için template olmayabilir, sorun değil
                continue

            for img_path in dir_path.glob("*.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[WARN] Template yüklenemedi: {img_path}")
                    continue

                w, h = img.shape[::-1]
                templates[direction_label].append((img, (w, h)))
                print(f"[INFO] Template yüklendi: {img_path}  -> yön: {direction_label}")

        # Hiç template yoksa patlat
        total_templates = sum(len(v) for v in templates.values())
        if total_templates == 0:
            raise RuntimeError(f"Hiç template bulunamadı: {self.templates_root}")

        print(f"[INFO] Toplam template sayısı: {total_templates}")
        return templates

    # -----------------------------
    # Aynı obje kontrolü
    # -----------------------------
    def _is_new_object(self, direction: str, center: tuple) -> bool:
        cx, cy = center
        for (d, (ox, oy)) in self.detected_objects:
            if d != direction:
                continue
            dist = math.dist((cx, cy), (ox, oy))
            if dist <= self.same_object_max_distance:
                # Aynı yönde ve konuma yakın → aynı obje
                return False
        return True

    # -----------------------------
    # Frame işle
    # -----------------------------
    def _detect_in_frame(self, frame_gray):
        """
        Frame içinde en iyi eşleşmeyi bulur.
        Dönüş:
            None  -> tespit yok
            dict  -> { 'direction', 'score', 'top_left', 'bottom_right', 'center' }
        """
        best_score = 0.0
        best_detection = None

        for direction, tmpl_list in self.templates.items():
            for template, (w, h) in tmpl_list:
                res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    center = (top_left[0] + w / 2.0, top_left[1] + h / 2.0)
                    best_detection = {
                        "direction": direction,
                        "score": max_val,
                        "top_left": top_left,
                        "bottom_right": bottom_right,
                        "center": center,
                    }

        if best_detection is not None and best_detection["score"] >= self.match_threshold:
            return best_detection
        return None

    # -----------------------------
    # Video işle
    # -----------------------------
    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")

        print(f"[INFO] Video açıldı: {video_path}")
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video bitti.")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detection = self._detect_in_frame(frame_gray)

            if detection is not None:
                direction = detection["direction"]
                score = detection["score"]
                tl = detection["top_left"]
                br = detection["bottom_right"]
                center = detection["center"]

                # Dikdörtgen çiz + yönü yaz
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    direction,
                    (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Yeni obje mi?
                if self._is_new_object(direction, center):
                    self.detected_objects.append((direction, center))

                    # Tüm tespitleri komut satırına yaz
                    print(
                        f"[DETECTION] Frame={frame_index:5d} | "
                        f"Yön={direction:6s} | "
                        f"Score={score:.3f} | "
                        f"Merkez=({center[0]:.1f}, {center[1]:.1f})"
                    )

                    # Video durdur, space'e basılınca devam et
                    cv2.imshow("Video", frame)
                    print(">> Tespit edildi. Devam etmek için SPACE (boşluk) tuşuna basın.")
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == self.pause_key_code or key == 27:  # space veya ESC
                            break
                        # yanlış tuşa basarsa yine bekle

            # Normal akışta göster
            cv2.imshow("Video", frame)
            key = cv2.waitKey(30) & 0xFF

            # ESC ile tamamen çık
            if key == 27:
                print("[INFO] ESC basıldı, program sonlandırılıyor.")
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

        print("\n[SUMMARY] Tespit edilen objeler:")
        for i, (direction, (cx, cy)) in enumerate(self.detected_objects, start=1):
            print(f"  {i:2d}. Yön={direction:6s}, Merkez=({cx:.1f}, {cy:.1f})")
