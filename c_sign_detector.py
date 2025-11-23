import cv2
import numpy as np
from pathlib import Path
import time
import math


class CSignDetector:
    """
    C işaretini (Sag / Sol / Yukari / Asagı) template matching ile tespit eder,
    videoyu oynatır ve her fiziksel obje için sadece ilk tespitte duraklatır.

    Özellikler:
    - Downscale ile hızlandırma (scale_factor)
    - Sadece her N framede bir ağır tespit (detect_every_n_frames)
    - Yeni obje başlatmak için yüksek skor eşiği (start_score_threshold)
    - Aynı fiziksel C'yi global olarak hatırlar, tekrar tespit etse bile
      ikinci kez [DETECTION] log'u ve pause yapmaz.
    """

    # Klasör adı -> Türkçe yön ismi
    DIR_LABELS = {
        "right": "Sag",
        "left": "Sol",
        "up": "Yukari",
        "down": "Asagı",
    }

    def __init__(
            self,
            templates_root: str,
            match_threshold: float = 0.6,
            same_object_max_distance: float = 80.0,  # global dedup için merkez mesafesi
            pause_key: str = " ",
            track_max_frame_gap: int = 300,
    ):
        self.templates_root = Path(templates_root)
        self.match_threshold = match_threshold

        # Yeni obje başlatmak için daha sert eşik (sadece güçlü tespitler obje sayılacak)
        self.start_score_threshold = max(self.match_threshold, 0.7)

        self.pause_key_code = ord(pause_key) if len(pause_key) == 1 else 32
        self.track_max_frame_gap = track_max_frame_gap

        # Aynı fiziksel C'yi global olarak tanımak için kullanılacak mesafe eşiği
        self.global_same_object_distance = same_object_max_distance

        # Hız için ölçekleme: 0.5 = hem yatay hem dikey boyutu yarıya indir (4x daha az iş)
        self.scale_factor = 0.5

        # Hız için: her kaç framede bir tespit yapılacak
        self.detect_every_n_frames = 3

        # Template'ler: { "Sag": [ (tmpl_img_scaled, (w_scaled, h_scaled)), ... ], ... }
        self.templates = self._load_templates()

        # O anda sahnede takip edilen tek obje (pause mantığı için)
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

        # Global olarak daha önce tespit ettiğimiz fiziksel C objeleri:
        # liste elemanı: (cx, cy)
        self.known_objects = []

    # -----------------------------
    # Template yükleme + scale (rotasyon YOK)
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
                    continue

                h, w = img.shape
                if self.scale_factor != 1.0:
                    new_w = max(1, int(w * self.scale_factor))
                    new_h = max(1, int(h * self.scale_factor))
                    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    templates[direction_label].append((scaled, (new_w, new_h)))
                else:
                    templates[direction_label].append((img, (w, h)))

        total_templates = sum(len(v) for v in templates.values())
        if total_templates == 0:
            raise RuntimeError(f"Hiç template bulunamadı: {self.templates_root}")

        return templates

    # -----------------------------
    # Template Matching (downscale frame üzerinde)
    # -----------------------------
    def _detect_in_frame(self, frame_gray):
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

        # Tüm template'ler üzerinde tara
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
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_raw = (direction, max_val, max_loc, (w_t, h_t))

        if best_raw is None or best_score < self.match_threshold:
            return None

        direction, score, top_left_small, (w_s, h_s) = best_raw

        # Küçük frame koordinatlarını orijinale geri ölçekle
        sx = sy = 1.0 / self.scale_factor if self.scale_factor != 1.0 else 1.0

        x1 = int(top_left_small[0] * sx)
        y1 = int(top_left_small[1] * sy)
        x2 = int((top_small_x := top_left_small[0] + w_s) * sx)
        y2 = int((top_small_y := top_left_small[1] + h_s) * sy)

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
    # Global aynı obje kontrolü
    # -----------------------------
    def _is_known_object(self, center):
        """Merkez noktası daha önce tespit edilmiş bir C'ye yakın mı? (yön dikkate alınmıyor)"""
        cx, cy = center
        for (kx, ky) in self.known_objects:
            dist = math.dist((cx, cy), (kx, ky))
            if dist <= self.global_same_object_distance:
                return True
        return False

    def _add_known_object(self, center):
        self.known_objects.append(center)

    # -----------------------------
    # Track yönetimi
    # -----------------------------
    def _start_new_track(self, detection, frame_index):
        """Yeni bir C objesi için track başlatır. Çizim veya pause yapmaz, global listeye ekler."""
        direction = detection["direction"]
        score = detection["score"]
        tl = detection["top_left"]
        br = detection["bottom_right"]
        center = detection["center"]

        self.active_track = {
            "direction": direction,  # Yön, ilk güçlü tespitte sabitlenir
            "first_frame": frame_index,
            "last_frame": frame_index,
            "last_center": center,
            "last_bbox": (tl, br),
            "first_score": score,
        }

        # Bu obje artık global olarak biliniyor
        self._add_known_object(center)

        # Ödevde istenen tek log
        print(
            f"[DETECTION] Frame={frame_index:5d} | "
            f"Yön={direction:6s} | "
            f"Score={score:.3f} | "
            f"Merkez=({center[0]:.1f}, {center[1]:.1f})"
        )

    def _update_active_track(self, detection, frame_index):
        # Yönü DEĞİŞTİRMİYORUZ, sadece bbox ve center güncelleniyor
        self.active_track["last_frame"] = frame_index
        self.active_track["last_center"] = detection["center"]
        self.active_track["last_bbox"] = (
            detection["top_left"],
            detection["bottom_right"],
        )

    def _finalize_active_track(self):
        self.active_track = None

    # -----------------------------
    # Video işle
    # -----------------------------
    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_index = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- Performans: sadece her N framede bir tespit ----
            detection = None
            if frame_index % self.detect_every_n_frames == 0:
                detection = self._detect_in_frame(frame_gray)

            # Bu frame için pause yapılacak mı?
            pause_requested = False

            # -----------------------------
            # TESPİT VARSA
            # -----------------------------
            if detection is not None:
                det_score = detection["score"]
                det_center = detection["center"]

                # Önce: bu obje global olarak önceden görülmüş mü?
                if self._is_known_object(det_center):
                    # Aynı fiziksel C daha önce loglandı → sadece çizebiliriz, log/pause yok
                    # Track yaratmadan direkt bbox çizeceğiz; aşağidaki çizim kısmı
                    pass
                else:
                    # Yeni bir fiziksel C olabilir
                    if self.active_track is None:
                        # Henüz hiç track yok → sadece güçlü skorlar yeni obje başlatır
                        if det_score >= self.start_score_threshold:
                            self._start_new_track(detection, frame_index)
                            pause_requested = True
                    else:
                        # Aktif track varsa → aynı fiziksel obje mi? (burada sadece zamana bakıyoruz)
                        last_frame = self.active_track["last_frame"]
                        frame_gap = frame_index - last_frame
                        same_object = frame_gap <= self.track_max_frame_gap

                        if same_object:
                            # Aynı track: yön sabit kalsın, bbox/center güncellensin
                            self._update_active_track(detection, frame_index)
                        else:
                            # Bu yeni bir C olabilir → yine güçlü skor şart
                            if det_score >= self.start_score_threshold:
                                self._finalize_active_track()
                                self._start_new_track(detection, frame_index)
                                pause_requested = True

            else:
                # Bu framede tespit yoksa ve track uzun süredir güncellenmiyorsa kapat
                if (
                        self.active_track is not None
                        and frame_index - self.active_track["last_frame"]
                        > self.track_max_frame_gap
                ):
                    self._finalize_active_track()

            # -----------------------------
            # ÇİZİM: aktif track varsa + C kaybolmamışsa
            # -----------------------------
            if self.active_track is not None:
                gap = frame_index - self.active_track["last_frame"]
                if gap <= self.detect_every_n_frames:
                    (x1, y1), (x2, y2) = self.active_track["last_bbox"]
                    ddir = self.active_track["direction"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        ddir,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # -------------------------
            # Overlay: Frame, yüzde, geçen süre (sadece görüntüde)
            # -------------------------
            elapsed = time.time() - start_time
            if total_frames > 0:
                percent = (frame_index + 1) / total_frames * 100.0
            else:
                percent = 0.0

            info_text = (
                f"Frame: {frame_index + 1}/{total_frames} "
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

            # -------------------------
            # GÖSTER + PAUSE LOGİĞİ
            # -------------------------
            cv2.imshow("Video", frame)

            if pause_requested:
                # Dikdörtgen çizilmiş frame üzerinde dur
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == self.pause_key_code:  # SPACE
                        break
                    if key == 27:  # ESC ile tamamen çık
                        cap.release()
                        cv2.destroyAllWindows()
                        self._finalize_active_track()
                        return
            else:
                key = cv2.waitKey(1) & 0xFF  # hızlı oynatma için 1 ms
                if key == 27:  # ESC
                    break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
        self._finalize_active_track()
