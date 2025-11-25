import cv2
from pathlib import Path
import math


class CSignDetector:
    DIR_LABELS = {
        "right": "Sag",
        "left": "Sol",
        "up": "Yukari",
        "down": "Asagi",
    }

    PRINT_LABELS = {
        "Sag": "Sağ",
        "Sol": "Sol",
        "Yukari": "Yukarı",
        "Asagi": "Aşağı",
    }

    def __init__(
            self,
            templates_root: str,
            match_threshold: float = 0.6,
            same_object_max_distance: float = 80.0,
            pause_key: str = " ",
            track_max_frame_gap: int = 300,
    ):
        self.templates_root = Path(templates_root)
        self.match_threshold = match_threshold

        self.start_score_threshold = max(self.match_threshold, 0.7)

        self.pause_key_code = ord(pause_key) if len(pause_key) == 1 else 32
        self.track_max_frame_gap = track_max_frame_gap

        self.global_same_object_distance = same_object_max_distance

        self.scale_factor = 0.5

        self.detect_every_n_frames = 3

        self.templates = self._load_templates()

        self.active_track = None

        self.known_objects = []

    def _load_templates(self):
        if not self.templates_root.exists():
            raise FileNotFoundError(f"Error: Template directory did not found.: {self.templates_root}")

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

        return templates

    def _detect_in_frame(self, frame_gray):

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
        best_raw = None

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

        sx = sy = 1.0 / self.scale_factor if self.scale_factor != 1.0 else 1.0

        x1 = int(top_left_small[0] * sx)
        y1 = int(top_left_small[1] * sy)
        x2 = int((top_small_x := top_left_small[0] + w_s) * sx)
        y2 = int((top_small_y := top_left_small[1] + h_s) * sy)

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

    def _is_known_object(self, center):
        cx, cy = center
        for (kx, ky) in self.known_objects:
            dist = math.dist((cx, cy), (kx, ky))
            if dist <= self.global_same_object_distance:
                return True
        return False

    def _add_known_object(self, center):
        self.known_objects.append(center)

    def _start_new_track(self, detection, frame_index):
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

        self._add_known_object(center)

        print_dir = self.PRINT_LABELS.get(direction, direction)
        print(
            f"[C] Frame={frame_index:5d} | "
            f"Direction={print_dir:6s} | "
            f"Score={score:.3f}"
        )

    def _update_active_track(self, detection, frame_index):
        self.active_track["last_frame"] = frame_index
        self.active_track["last_center"] = detection["center"]
        self.active_track["last_bbox"] = (
            detection["top_left"],
            detection["bottom_right"],
        )

    def _finalize_active_track(self):
        self.active_track = None
