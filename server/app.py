# ============= #
# server/app.py #
# ============= #

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe Hands topology for 21 landmarks.
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).resolve().parent / "model" / "gesture_recognizer.task"
    parser = argparse.ArgumentParser(
        description="Rozpoznawanie ukladu dloni i gestow z obrazu kamery (MediaPipe Tasks)."
    )
    parser.add_argument("--model", type=Path, default=default_model, help="Sciezka do pliku .task")
    parser.add_argument("--camera-id", type=int, default=0, help="Indeks kamery (domyslnie 0)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimalna pewnosc rozpoznanego gestu (0-1)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maksymalna liczba sledzonych dloni",
    )
    return parser.parse_args()


def create_gesture_recognizer(model_path: Path, max_hands: int):
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=max_hands,
    )
    return vision.GestureRecognizer.create_from_options(options)


def draw_landmarks(frame_bgr, hand_landmarks: Sequence) -> None:
    height, width = frame_bgr.shape[:2]
    points: List[Tuple[int, int]] = []

    for lm in hand_landmarks:
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        points.append((x_px, y_px))
        cv2.circle(frame_bgr, (x_px, y_px), 3, (60, 220, 60), -1)

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame_bgr, points[start_idx], points[end_idx], (255, 210, 80), 2)


def put_overlay_text(frame_bgr, line1: str, line2: str) -> None:
    cv2.rectangle(frame_bgr, (10, 10), (560, 90), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        line1,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        line2,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )


def open_camera(camera_id: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_id)
    return capture


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        print(f"[ERROR] Nie znaleziono modelu: {args.model}")
        print("Podaj poprawna sciezke parametrem --model")
        return 1

    try:
        recognizer = create_gesture_recognizer(args.model, args.max_hands)
    except Exception as ex:
        print("[ERROR] Nie udalo sie uruchomic GestureRecognizer.")
        print(f"Szczegoly: {ex}")
        return 1

    cap = open_camera(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Nie mozna otworzyc kamery o indeksie {args.camera_id}")
        recognizer.close()
        return 1

    window_name = "Zdalne Droidy - Rozpoznawanie dloni"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    previous_time = time.perf_counter()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[WARN] Brak klatki z kamery, koncze.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = recognizer.recognize(mp_image)

            gesture_text = "Gesture: brak"
            hand_text = "Hands: 0"

            if result.hand_landmarks:
                hand_text = f"Hands: {len(result.hand_landmarks)}"
                for hand_lms in result.hand_landmarks:
                    draw_landmarks(frame_bgr, hand_lms)

            if result.gestures and result.gestures[0]:
                best = result.gestures[0][0]
                if best.score >= args.min_score:
                    gesture_text = f"Gesture: {best.category_name} ({best.score:.2f})"
                else:
                    gesture_text = f"Gesture: ponizej progu ({best.score:.2f})"

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            put_overlay_text(frame_bgr, gesture_text, f"{hand_text} | FPS: {fps:.1f} | Wyjscie [Q]")
            cv2.imshow(window_name, frame_bgr)

            # End loop if the window was closed from the title bar (X button).
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        recognizer.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
