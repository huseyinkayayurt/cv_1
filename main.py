import argparse
import cv2
import time

from c_sign_detector import CSignDetector
from hazmat import HazmatDetector


def parse_args():
    parser = argparse.ArgumentParser(description="CV homework 1")
    parser.add_argument("video")
    parser.add_argument("--c-templates-root", default="templates/c_sign")
    parser.add_argument("--hazmat-templates-root", default="templates/hazmat")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--pause-distance", type=float, default=80.0, )
    return parser.parse_args()


def process_video_combined(video_path, c_detector, hazmat_detector, pause_key=" "):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video cannot be opened: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pause_key_code = ord(pause_key) if len(pause_key) == 1 else 32

    frame_index = 0
    start_time = time.time()

    # cv2.namedWindow("Combined Detection", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Combined Detection", 1024, 768)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pause_requested = False

        c_detection = None
        if frame_index % c_detector.detect_every_n_frames == 0:
            c_detection = c_detector._detect_in_frame(frame_gray)

        if c_detection is not None:
            det_score = c_detection["score"]
            det_center = c_detection["center"]

            if c_detector._is_known_object(det_center):
                pass
            else:
                if c_detector.active_track is None:
                    if det_score >= c_detector.start_score_threshold:
                        c_detector._start_new_track(c_detection, frame_index)
                        pause_requested = True
                else:
                    last_frame = c_detector.active_track["last_frame"]
                    frame_gap = frame_index - last_frame
                    same_object = frame_gap <= c_detector.track_max_frame_gap

                    if same_object:
                        c_detector._update_active_track(c_detection, frame_index)
                    else:
                        if det_score >= c_detector.start_score_threshold:
                            c_detector._finalize_active_track()
                            c_detector._start_new_track(c_detection, frame_index)
                            pause_requested = True
        else:
            if (
                    c_detector.active_track is not None
                    and frame_index - c_detector.active_track["last_frame"]
                    > c_detector.track_max_frame_gap
            ):
                c_detector._finalize_active_track()

        if c_detector.active_track is not None:
            gap = frame_index - c_detector.active_track["last_frame"]
            if gap <= c_detector.detect_every_n_frames:
                (x1, y1), (x2, y2) = c_detector.active_track["last_bbox"]
                ddir = c_detector.active_track["direction"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(
                    frame,
                    ddir,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

        hazmat_detections, hazmat_pause, _ = hazmat_detector.detect_in_frame(frame, frame_index)

        if hazmat_pause:
            pause_requested = True

        for det in hazmat_detections:
            x, y, w, h = det["box"]
            label = f"{det['name']} ({det['score']:.2f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + tw, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        elapsed = time.time() - start_time
        if total_frames > 0:
            percent = (frame_index + 1) / total_frames * 100.0
        else:
            percent = 0.0

        info_text = (
            f"Frame: {frame_index + 1}/{total_frames} "
            f"({percent:4.1f}%)  Time: {elapsed:5.1f}s"
        )

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            info_text,
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Detection", frame)

        if pause_requested:
            overlay = frame.copy()
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.imshow("Detection", frame)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == pause_key_code:  # SPACE
                    break
                if key == 27:  # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    c_detector = CSignDetector(
        templates_root=args.c_templates_root,
        match_threshold=args.threshold,
        same_object_max_distance=args.pause_distance,
        pause_key=" "
    )

    hazmat_detector = HazmatDetector(args.hazmat_templates_root)

    process_video_combined(
        args.video,
        c_detector,
        hazmat_detector,
        pause_key=" "
    )


if __name__ == "__main__":
    main()
