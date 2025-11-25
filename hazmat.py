import cv2
import numpy as np
import os
import sys
from skimage.feature import hog

# --- AYARLAR ---
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
IMG_SIZE = (64, 64)

# GENEL EŞİK (Varsayılan)
# Renkli ve belirgin işaretler için bu değer kullanılır.
DEFAULT_SIMILARITY_THRESHOLD = 0.65

# --- KRİTİK AYAR: SINIF BAZLI EŞİK DEĞERLERİ ---
# Beyaz arka planlı veya karışmaya müsait işaretler için eşiği yükseltiyoruz.
# Böylece duvardaki beyaz lekeyi "Poison" sanmaz.
CUSTOM_THRESHOLDS = {
    "poison": 0.65,  # Beyaz olduğu için sıkı kontrol
    "inhalation-hazard": 0.65,  # Beyaz
    "radioactive": 0.65,  # Sarı-Beyaz, karışabilir
    "corrosive": 0.65,  # Beyaz-Siyah
    "organic-peroxide": 0.60,  # Sarı/Kırmızı ama karışık
    "oxidizer": 0.60,  # Sarı
    "spontaneously-combustible": 0.65,
    "flammable-gas": 0.65,
    "flammable-solid": 0.65

    # Diğerleri (Explosives, Flammable Gas vb.) 0.50 ile devam eder.
}

TRACKING_DISTANCE_LIMIT = 250
MEMORY_TIMEOUT = 150

HAZMAT_NAMES = [
    "explosives", "blasting-agents", 'flammable-gas', 'non-flammable-gas',
    'oxygen', 'fuel-oil', 'dangerous-when-wet', 'flammable-solid',
    'spontaneously-combustible', 'oxidizer', 'organic-peroxide',
    'inhalation-hazard', 'poison', 'radioactive', 'corrosive'
]

# --- RENK TANIMLAMALARI ---
HAZMAT_COLORS = {
    "explosives": ["orange"],
    "blasting-agents": ["orange"],
    "flammable-gas": ["red"],
    "non-flammable-gas": ["green"],
    "oxygen": ["yellow"],
    "fuel-oil": ["red", "black"],
    "dangerous-when-wet": ["blue"],
    "flammable-solid": ["red", "white"],
    "spontaneously-combustible": ["red", "white"],
    "oxidizer": ["yellow"],
    "organic-peroxide": ["red", "yellow"],
    "inhalation-hazard": ["white"],
    "poison": ["white"],
    "radioactive": ["white", "yellow"],
    "corrosive": ["white", "black"]
}

VIDEO_FILE = "data/odev1.mp4"
TEMPLATE_FOLDER = "templates/hazmat"


class HazmatDetector:
    def __init__(self, template_dir):
        self.templates_hog = {}
        self.load_templates(template_dir)
        self.known_objects = []

    def preprocess_hog_input(self, img):
        img = cv2.resize(img, IMG_SIZE)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def get_hog_features(self, img_gray):
        features = hog(img_gray,
                       orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       block_norm='L2-Hys',
                       visualize=False)
        return features

    def load_templates(self, template_dir):
        print(f"[BILGI] Şablonlar yükleniyor: {template_dir}")
        if not os.path.exists(template_dir):
            print("HATA: Klasör bulunamadı.")
            return

        for name in HAZMAT_NAMES:
            path = os.path.join(template_dir, f"{name}.png")
            if not os.path.exists(path):
                path = os.path.join(template_dir, f"{name}.jpg")
            if not os.path.exists(path): continue

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

            if img.shape[2] == 4:
                trans_mask = img[:, :, 3] == 0
                img[trans_mask] = [255, 255, 255, 255]
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img_gray = self.preprocess_hog_input(img)
            features = self.get_hog_features(img_gray)

            norm = np.linalg.norm(features)
            if norm != 0: features /= norm

            self.templates_hog[name] = features
        print(f"[BILGI] {len(self.templates_hog)} şablon hazır.")

    def get_dominant_color_category(self, roi):
        """
        Geliştirilmiş Renk Tespiti.
        Beyaz rengi daha hassas algılamak için Value ve Saturation kontrolü.
        """
        roi_small = cv2.resize(roi, (32, 32))
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)

        # Merkeze odaklan
        h, w = hsv.shape[:2]
        center_hsv = hsv[h // 4:3 * h // 4, w // 4:3 * w // 4]

        mean_h = np.mean(center_hsv[:, :, 0])
        mean_s = np.mean(center_hsv[:, :, 1])
        mean_v = np.mean(center_hsv[:, :, 2])

        # Beyaz tespiti (Düşük Doygunluk, Yüksek Parlaklık)
        # Saturation < 50 VE Value > 130 (Gri duvarları elemek için Value yüksek olmalı)
        if mean_s < 50 and mean_v > 130:
            return "white"

        # Siyah tespiti (Çok düşük parlaklık)
        if mean_v < 60:
            return "black"

        # Renkler
        if (mean_h >= 0 and mean_h <= 10) or (mean_h >= 160 and mean_h <= 180):
            return "red"
        elif 11 <= mean_h <= 25:
            return "orange"
        elif 26 <= mean_h <= 35:
            return "yellow"
        elif 36 <= mean_h <= 85:
            return "green"
        elif 86 <= mean_h <= 130:
            return "blue"

        return "unknown"

    def verify_color(self, roi, hazmat_name):
        detected_color = self.get_dominant_color_category(roi)
        expected_colors = HAZMAT_COLORS.get(hazmat_name, [])

        # Tam eşleşme
        if detected_color in expected_colors:
            return True

        # Özel Durum: Sarı işaretler bazen beyaz veya turuncu algılanabilir (ışık yüzünden)
        if "yellow" in expected_colors and (detected_color == "white" or detected_color == "orange"):
            return True

        # Özel Durum: Beyaz işaretler bazen çok parlaksa sarımsı çıkabilir
        if "white" in expected_colors and detected_color == "yellow":
            # Ama eğer "Poison" (Beyaz) ise ve renk bariz Sarı ise reddetmeliyiz.
            # Burada threshold yüksek olduğu için riske girmeyip reddedelim.
            # Sadece 'radioactive' hem beyaz hem sarı içerir, o geçer.
            if hazmat_name == "poison" or hazmat_name == "inhalation-hazard":
                return False
            return True

        return False

    def filter_contained_boxes(self, candidates):
        if not candidates: return []
        keep_indices = [True] * len(candidates)

        for i in range(len(candidates)):
            xi, yi, wi, hi = candidates[i][:4]
            for j in range(len(candidates)):
                if i == j: continue
                xj, yj, wj, hj = candidates[j][:4]
                if (xi <= xj) and (yi <= yj) and (xi + wi >= xj + wj) and (yi + hi >= yj + hj):
                    keep_indices[i] = False

        final_candidates = []
        for k, keep in enumerate(keep_indices):
            if keep: final_candidates.append(candidates[k])
        return final_candidates

    def detect_in_frame(self, frame, frame_count):
        detections = []
        should_pause = False
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 150)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800 or area > (frame_area * 0.30): continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                if 0.75 <= aspect_ratio <= 1.25:
                    if x < 5 or y < 5 or (x + w) > frame_w - 5 or (y + h) > frame_h - 5: continue
                    roi = frame[y:y + h, x:x + w]

                    # Duvar elemesi (Düz renk kontrolü)
                    if np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) < 35: continue

                    candidates.append((x, y, w, h, roi))

        candidates = self.filter_contained_boxes(candidates)
        current_frame_objects = []

        for (x, y, w, h, roi) in candidates:
            try:
                roi_processed = self.preprocess_hog_input(roi)
                roi_hog = self.get_hog_features(roi_processed)
                norm = np.linalg.norm(roi_hog)
                if norm == 0: continue
                roi_hog /= norm

                matches = []
                for name, template_hog in self.templates_hog.items():
                    # 1. Her sınıf için kendi eşik değerini kullan
                    threshold = CUSTOM_THRESHOLDS.get(name, DEFAULT_SIMILARITY_THRESHOLD)
                    threshold = DEFAULT_SIMILARITY_THRESHOLD

                    score = np.dot(roi_hog, template_hog)

                    if score > threshold:
                        matches.append((score, name))

                matches.sort(key=lambda x: x[0], reverse=True)

                final_name = None
                final_score = 0

                for score, name in matches[:3]:
                    # 2. Renk Doğrulaması
                    if self.verify_color(roi, name):
                        final_name = name
                        final_score = score
                        break

                if final_name:
                    centroid = (x + w // 2, y + h // 2)
                    obj_data = {
                        "name": final_name,
                        "score": final_score,
                        "box": (x, y, w, h),
                        "center": centroid
                    }
                    current_frame_objects.append(obj_data)
                    detections.append(obj_data)

            except Exception:
                pass

        # Takip Mantığı
        for obj in current_frame_objects:
            is_new_object = True
            for known in self.known_objects:
                if known['name'] == obj['name']:
                    dist = np.sqrt((known['center'][0] - obj['center'][0]) ** 2 +
                                   (known['center'][1] - obj['center'][1]) ** 2)
                    if dist < TRACKING_DISTANCE_LIMIT:
                        known['center'] = obj['center']
                        known['last_seen'] = frame_count
                        is_new_object = False
                        break

            if is_new_object:
                should_pause = True
                self.known_objects.append({
                    'name': obj['name'],
                    'center': obj['center'],
                    'last_seen': frame_count
                })
                print(f">>> YENİ TESPİT: {obj['name']} | Skor: {obj['score']:.2f} | Frame: {frame_count}")

        self.known_objects = [obj for obj in self.known_objects if (frame_count - obj['last_seen']) < MEMORY_TIMEOUT]
        return detections, should_pause, dilated_edges


def main(video_path, template_dir):
    if not os.path.exists(video_path):
        print(f"HATA: Video yok -> {video_path}")
        return
    if not os.path.exists(template_dir):
        print(f"HATA: Template yok -> {template_dir}")
        return

    detector = HazmatDetector(template_dir)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video açılamadı.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    frame_counter = 0
    cv2.namedWindow("Hazmat Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hazmat Detection", 800, 600)

    print(f"\n--- BAŞLIYOR ---")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_counter += 1

        detections, should_pause, _ = detector.detect_in_frame(frame, frame_counter)

        current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        elapsed_seconds = current_time_msec / 1000.0
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        percentage = (frame_counter / total_frames) * 100

        for det in detections:
            x, y, w, h = det["box"]
            label = f"{det['name']} ({det['score']:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + tw, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        info_text = f"Frame: {frame_counter}/{total_frames} | %{percentage:.1f} | {mins:02d}:{secs:02d}"
        cv2.putText(frame, info_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hazmat Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        if should_pause:
            overlay = frame.copy()
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.imshow("Hazmat Detection", frame)
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    v_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_FILE
    t_path = sys.argv[2] if len(sys.argv) > 2 else TEMPLATE_FOLDER
    main(v_path, t_path)
