import cv2
import numpy as np
import os
from skimage.feature import hog

HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
IMG_SIZE = (64, 64)

DEFAULT_SIMILARITY_THRESHOLD = 0.66

CUSTOM_THRESHOLDS = {
    "poison": 0.65,
    "inhalation-hazard": 0.65,
    "radioactive": 0.70,
    "corrosive": 0.60,
    "organic-peroxide": 0.7,
    "oxidizer": 0.60,
    "spontaneously-combustible": 0.65,
    "flammable-gas": 0.65,
    "flammable-solid": 0.65
}

TRACKING_DISTANCE_LIMIT = 250
MEMORY_TIMEOUT = 150

HAZMAT_NAMES = [
    "explosives", "blasting-agents", 'flammable-gas', 'non-flammable-gas',
    'oxygen', 'fuel-oil', 'dangerous-when-wet', 'flammable-solid',
    'spontaneously-combustible', 'oxidizer', 'organic-peroxide',
    'inhalation-hazard', 'poison', 'radioactive', 'corrosive'
]

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
        if not os.path.exists(template_dir):
            print("Error: Template directory cannot be found.")
            return

        for name in HAZMAT_NAMES:
            path = os.path.join(template_dir, f"{name}.png")
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

    def get_dominant_color_category(self, roi):
        roi_small = cv2.resize(roi, (32, 32))
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)

        h, w = hsv.shape[:2]
        center_hsv = hsv[h // 4:3 * h // 4, w // 4:3 * w // 4]

        mean_h = np.mean(center_hsv[:, :, 0])
        mean_s = np.mean(center_hsv[:, :, 1])
        mean_v = np.mean(center_hsv[:, :, 2])

        if mean_s < 50 and mean_v > 130:
            return "white"

        if mean_v < 60:
            return "black"

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

        if detected_color in expected_colors:
            return True

        if "yellow" in expected_colors and (detected_color == "white" or detected_color == "orange"):
            return True

        if "white" in expected_colors and detected_color == "yellow":
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
                    threshold = CUSTOM_THRESHOLDS.get(name, DEFAULT_SIMILARITY_THRESHOLD)
                    score = np.dot(roi_hog, template_hog)

                    if score > threshold:
                        matches.append((score, name))

                matches.sort(key=lambda x: x[0], reverse=True)

                final_name = None
                final_score = 0

                for score, name in matches[:3]:
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
                print(f"[H] Frame={frame_count:5d} | "
                      f"Name={obj['name']} | "
                      f"Score={obj['score']:.2f}")

        self.known_objects = [obj for obj in self.known_objects if (frame_count - obj['last_seen']) < MEMORY_TIMEOUT]
        return detections, should_pause, dilated_edges
