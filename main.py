import argparse
from c_sign_detector import CSignDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="C işaretlerini videoda tespit edip yönünü belirleyen program."
    )
    parser.add_argument(
        "video",
        help="Girdi video dosyasının yolu (ör: input.mp4)"
    )
    parser.add_argument(
        "--templates-root",
        default="templates/c_sign",
        help="C işareti template kök klasörü (varsayılan: templates/c_sign)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Template eşleşme eşiği (0–1 arası, varsayılan: 0.6)"
    )
    parser.add_argument(
        "--pause-distance",
        type=float,
        default=80.0,
        help="Aynı objeyi tanımak için merkezler arası maksimum mesafe (px, varsayılan: 60)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    detector = CSignDetector(
        templates_root=args.templates_root,
        match_threshold=args.threshold,
        same_object_max_distance=args.pause_distance,
        pause_key=" "  # space tuşu, istersen burayı başka char yapabilirsin
    )

    detector.process_video(args.video)


if __name__ == "__main__":
    main()
