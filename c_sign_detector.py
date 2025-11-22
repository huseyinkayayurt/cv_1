import cv2
import numpy as np
from pathlib import Path


class CSignDetector:
    """
    C işaretini (Sağ / Sol / Yukarı / Aşağı) template matching ile tespit eder,
    videoyu oynatır ve her fiziksel obje için sadece ilk tespitte duraklatır.
    Ek olarak frameleri ve yüzdeyi ekrana yazar.
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
        same_object_max_distance: float = 60.0,  # artık kullanılmıyor ama param bozulmasın diye duruyor
        pause_key: str = " ",
        track_max_frame_gap: int = 120,  # aynı C için uzun süre tolerans (frame cinsinden)
    ):
        self.templates_root = Path(templates_root)
        self.match_threshold = match_threshold
        self.pause_key_code = ord(pause_key) if len(pause_key) == 1 else 32
        self.track_max_frame_gap = track_max_frame_gap

        # { "Sağ": [ (template_img, (w, h)), ... ], ... }
        self.templates = self._load_templates()

        # Bitmiş track'ler (rapor için)
        self.finished_tracks = []

        # Şu an sahnede takip edilen tek obje
        # None veya dict:
        # {
        #   'direction': str,
        #   'first_frame': int,
        #   'last_frame': int,
        #   'last_center': (cx, cy),
        #   'last_bbox': ((x1, y1), (x2, y2)),
        #   'first_score': float
        # }
        self.active_track = None

        # HSV altyapısı (şu an detection'ı boğmayacak şekilde çok geniş aralık)
        # İstersen bunları daraltarak gerçek renk filtresine çevirebilirsin.
        self.hsv_lower = np.array([0, 0, 0], dtype=np.uint8)
        self.hsv_upper = np.array([180, 255, 255], dtype=np.uint8)

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
                continue

            for img_path in dir_path.glob("*.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[WARN] Template yüklenemedi: {img_path}")
                    continue

                w, h = img.shape[::-1]
                templates[direction_label].append((img, (w, h)))
                print(f"[INFO] Template yüklendi: {img_path}  -> yön: {direction_label}")

        total_templates = sum(len(v) for v in templates.values())
        if total_templates == 0:
            raise RuntimeError(f"Hiç template bulunamadı: {self.templates_root}")

        print(f"[INFO] Toplam template sayısı: {total_templates}")
        return templates

    # -----------------------------
    # Template Matching (+ hafif HSV check)
    # -----------------------------
    def _detect_in_frame(self, frame_bgr, frame_gray):
        """
        Frame içinde en iyi eşleşmeyi bulur.
        Dönüş:
            None  -> tespit yok
            dict  -> { 'direction', 'score', 'top_left', 'bottom_right', 'center' }
        """

        # HSV hesapla (ileride istersen daraltıp gerçek renk filtresi yapabilirsin)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        best_score = 0.0
        best_detection = None

        for direction, tmpl_list in self.templates.items():
            for template, (w, h) in tmpl_list:
                # Eski, sağlam hali: full gray frame üzerinde matchTemplate
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

        if best_detection is None or best_detection["score"] < self.match_threshold:
            return None

        # Hafif HSV filtresi: tespit edilen kutudaki mask oranına bak
        tl = best_detection["top_left"]
        br = best_detection["bottom_right"]
        x1, y1 = tl
        x2, y2 = br
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame_gray.shape[1] - 1)
        y2 = min(y2, frame_gray.shape[0] - 1)

        roi = mask[y1:y2, x1:x2]
        area = roi.size if roi.size > 0 else 1
        coverage = cv2.countNonZero(roi) / float(area)

        # coverage threshold'ünü çok düşük tutuyorum ki gerçek tespitler kaçmasın
        if coverage < 0.01:
            return None

        return best_detection

    # -----------------------------
    # Track yönetimi
    # -----------------------------
    def _start_new_track(self, detection, frame_index, frame_to_show):
        direction = detection["direction"]
        score = detection["score"]
        tl = detection["top_left"]
        br = detection["bottom_right"]
        center = detection["center"]

        self.active_track = {
            "direction": direction,
            "first_frame": frame_index,
            "last_frame": frame_index,
            "last_center": center,
            "last_bbox": (tl, br),
            "first_score": score,
        }

        # Konsola yazdır
        print(
            f"[DETECTION] Frame={frame_index:5d} | "
            f"Yön={direction:6s} | "
            f"Score={score:.3f} | "
            f"Merkez=({center[0]:.1f}, {center[1]:.1f})"
        )

        # İlk kez görülen obje için video durdurma
        cv2.imshow("Video", frame_to_show)
        print(">> Tespit edildi. Devam etmek için SPACE (boşluk) tuşuna basın.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == self.pause_key_code or key == 27:  # space veya ESC
                break

    def _update_active_track(self, detection, frame_index):
        self.active_track["last_frame"] = frame_index
        self.active_track["last_center"] = detection["center"]
        self.active_track["last_bbox"] = (
            detection["top_left"],
            detection["bottom_right"],
        )

    def _finalize_active_track(self):
        if self.active_track is not None:
            self.finished_tracks.append(self.active_track)
            self.active_track = None

    # -----------------------------
    # Video işle
    # -----------------------------
    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Video açıldı: {video_path}")
        print(f"[INFO] Toplam frame: {total_frames}")

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video bitti.")
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detection = self._detect_in_frame(frame, frame_gray)

            if detection is not None:
                direction = detection["direction"]
                tl = detection["top_left"]
                br = detection["bottom_right"]
                center = detection["center"]

                # Dikdörtgen + yön yaz
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

                if self.active_track is None:
                    # Hiç track yoksa -> yeni obje
                    self._start_new_track(detection, frame_index, frame.copy())
                else:
                    # Aktif track varsa: aynı obje mi?
                    last_frame = self.active_track["last_frame"]
                    last_direction = self.active_track["direction"]
                    frame_gap = frame_index - last_frame

                    # Yön aynı ve aradaki frame aralığı makul ise aynı obje kabul et
                    if (
                        direction == last_direction
                        and frame_gap <= self.track_max_frame_gap
                    ):
                        self._update_active_track(detection, frame_index)
                    else:
                        # Önceki objeyi kapat, yeni obje başlat
                        self._finalize_active_track()
                        self._start_new_track(detection, frame_index, frame.copy())
            else:
                # Bu framede C yoksa ve uzun süredir göremiyorsak track'i kapat
                if (
                    self.active_track is not None
                    and frame_index - self.active_track["last_frame"]
                    > self.track_max_frame_gap
                ):
                    self._finalize_active_track()

            # -------------------------
            # Overlay: Frame ve yüzde
            # -------------------------
            if total_frames > 0:
                percent = (frame_index + 1) / total_frames * 100.0
                info_text = f"Frame: {frame_index+1} / {total_frames} ({percent:4.1f}%)"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Normal akışta göster
            cv2.imshow("Video", frame)
            key = cv2.waitKey(1) & 0xFF  # daha akıcı oynatma için 1 ms

            if key == 27:  # ESC
                print("[INFO] ESC basıldı, program sonlandırılıyor.")
                break

            frame_index += 1

        # Video kapat
        cap.release()
        cv2.destroyAllWindows()

        # Son track açık kalmışsa kapat
        self._finalize_active_track()

        print("\n[SUMMARY] Tespit edilen objeler:")
        for i, t in enumerate(self.finished_tracks, start=1):
            bbox = t["last_bbox"]
            (x1, y1), (x2, y2) = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            print(
                f"  {i:2d}. Yön={t['direction']:6s}, "
                f"İlk frame={t['first_frame']}, Son frame={t['last_frame']}, "
                f"Merkez=({cx:.1f}, {cy:.1f})"
            )
