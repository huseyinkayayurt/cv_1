import cv2
import numpy as np
from pathlib import Path
import time


class CSignDetector:
    """
    C işaretini (Sağ / Sol / Yukarı / Aşağı) template matching ile tespit eder,
    videoyu oynatır ve her fiziksel obje için sadece ilk tespitte duraklatır.
    - Açılı C'ler için: her template'in döndürülmüş versiyonlarını kullanır.
    - Hız için: frame'leri downscale ederek çalışır.
    - Overlay: frame numarası, yüzde ve geçen süreyi ekrana yazar.
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
        same_object_max_distance: float = 60.0,  # artık kullanılmıyor ama argüman bozulmasın diye bırakıldı
        pause_key: str = " ",
        track_max_frame_gap: int = 120,
    ):
        self.templates_root = Path(templates_root)
        self.match_threshold = match_threshold
        self.pause_key_code = ord(pause_key) if len(pause_key) == 1 else 32
        self.track_max_frame_gap = track_max_frame_gap

        # Hız için ölçekleme: 0.5 = hem yatay hem dikey boyutu yarıya indir (4x daha az iş)
        self.scale_factor = 0.5

        # Template'ler: { "Sağ": [ (tmpl_img_scaled, (w_scaled, h_scaled)), ... ], ... }
        self.templates = self._load_templates()

        # Bitmiş track'ler (rapor için)
        self.finished_tracks = []

        # Şu an sahnede takip edilen tek obje
        self.active_track = None
        # active_track:
        # {
        #   'direction': str,
        #   'first_frame': int,
        #   'last_frame': int,
        #   'last_center': (cx, cy),
        #   'last_bbox': ((x1, y1), (x2, y2)),
        #   'first_score': float
        # }

    # -----------------------------
    # Template yükleme + rotasyon + scale
    # -----------------------------
    def _generate_rotated_and_scaled(self, img_gray):
        """
        Verilen gri template için birkaç açı (0, ±15 derece) üreterek
        her birini scale_factor ile yeniden boyutlandırır.
        """
        angles = [0, -15, 15]  # Gerekirse listeye -25, 25 de eklenebilir.
        h, w = img_gray.shape
        center = (w // 2, h // 2)

        out = []

        for angle in angles:
            if angle == 0:
                rotated = img_gray
            else:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    img_gray,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,  # köşe artefaktlarını azalt
                )

            if self.scale_factor != 1.0:
                new_w = max(1, int(w * self.scale_factor))
                new_h = max(1, int(h * self.scale_factor))
                rotated_scaled = cv2.resize(
                    rotated, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                out.append((rotated_scaled, (new_w, new_h)))
            else:
                out.append((rotated, (w, h)))

        return out

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

                rotated_scaled_list = self._generate_rotated_and_scaled(img)
                templates[direction_label].extend(rotated_scaled_list)
                print(
                    f"[INFO] Template yüklendi: {img_path}  -> yön: {direction_label} "
                    f"(rotated x{len(rotated_scaled_list)})"
                )

        total_templates = sum(len(v) for v in templates.values())
        if total_templates == 0:
            raise RuntimeError(f"Hiç template bulunamadı: {self.templates_root}")

        print(f"[INFO] Toplam efektif template sayısı (rotated dahil): {total_templates}")
        return templates

    # -----------------------------
    # Template Matching (downscale frame üzerinde)
    # -----------------------------
    def _detect_in_frame(self, frame_bgr, frame_gray):
        """
        Frame içinde en iyi eşleşmeyi bulur.
        Dönüş:
            None  -> tespit yok
            dict  -> { 'direction', 'score', 'top_left', 'bottom_right', 'center' }
            (Koordinatlar ORİJİNAL frame boyutlarına göre döner.)
        """
        # Frame'i downscale et
        if self.scale_factor != 1.0:
            small_gray = cv2.resize(
                frame_gray,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small_gray = frame_gray

        best_score = 0.0
        best_raw = None  # (direction, max_val, top_left_small, (w_small, h_small))

        # Tüm (döndürülmüş + scale edilmiş) template'ler üzerinde tara
        for direction, tmpl_list in self.templates.items():
            for template, (w_t, h_t) in tmpl_list:
                if (
                    small_gray.shape[1] < w_t
                    or small_gray.shape[0] < h_t
                    or w_t <= 0
                    or h_t <= 0
                ):
                    continue

                res = cv2.matchTemplate(small_gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_raw = (direction, max_val, max_loc, (w_t, h_t))

        if best_raw is None or best_score < self.match_threshold:
            return None

        direction, score, top_left_small, (w_s, h_s) = best_raw

        # Küçük frame koordinatlarını orijinale geri ölçekle
        if self.scale_factor != 1.0:
            sx = 1.0 / self.scale_factor
            sy = 1.0 / self.scale_factor
        else:
            sx = sy = 1.0

        x1 = int(top_left_small[0] * sx)
        y1 = int(top_left_small[1] * sy)
        x2 = int((top_left_small[0] + w_s) * sx)
        y2 = int((top_left_small[1] + h_s) * sy)

        # Bounds içinde tut
        h_frame, w_frame = frame_gray.shape
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame - 1))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame - 1))

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        detection = {
            "direction": direction,
            "score": score,
            "top_left": (x1, y1),
            "bottom_right": (x2, y2),
            "center": (cx, cy),
        }
        return detection

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
        start_time = time.time()

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
                    # Aktif track varsa: yön aynı ve aradaki frame aralığı makulse = aynı obje
                    last_frame = self.active_track["last_frame"]
                    last_direction = self.active_track["direction"]
                    frame_gap = frame_index - last_frame

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
            # Overlay: Frame, yüzde, geçen süre
            # -------------------------
            elapsed = time.time() - start_time
            if total_frames > 0:
                percent = (frame_index + 1) / total_frames * 100.0
            else:
                percent = 0.0

            info_text = (
                f"Frame: {frame_index+1}/{total_frames} "
                f"({percent:4.1f}%)  Time: {elapsed:5.1f}s"
            )
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Her 200 framede bir konsola da yaz
            if frame_index % 200 == 0:
                print(
                    f"[PROGRESS] Frame {frame_index+1}/{total_frames} "
                    f"({percent:4.1f}%)  Elapsed: {elapsed:5.1f}s"
                )

            # Normal akışta göster
            cv2.imshow("Video", frame)
            key = cv2.waitKey(1) & 0xFF  # hızlı oynatma için 1 ms

            if key == 27:  # ESC
                print("[INFO] ESC basıldı, program sonlandırılıyor.")
                break

            frame_index += 1

        # Video kapat
        cap.release()
        cv2.destroyAllWindows()

        # Son track açık kalmışsa kapat
        self._finalize_active_track()

        total_elapsed = time.time() - start_time
        print(f"\n[INFO] Toplam süre: {total_elapsed:.2f} saniye")

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
