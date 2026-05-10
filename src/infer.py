from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

from detector import PadelObjectDetector
from shot_classifier import ShotClassifier
from tracker import BallTracker


VIDEO_PATH = r"data\raw\inference_sample_video.mp4"
WEIGHTS_PATH = r"runs\detect\models\padel_detector\weights\best.pt"

OUTPUT_DIR = Path("outputs")
OUTPUT_VIDEO = OUTPUT_DIR / "annotated_inference.mp4"
OUTPUT_JSON = OUTPUT_DIR / "shots.json"
OUTPUT_CSV = OUTPUT_DIR / "shots.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.json"

CONF = 0.25
IOU = 0.50
IMGSZ = 768
DISPLAY_WIDTH = 1200


def resize_for_display(frame, target_width=1200):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def draw_detections(frame, players, rackets, tracked_ball, event=None):
    for player in players:
        x1, y1, x2, y2 = map(int, player["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "Player",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    for racket in rackets:
        x1, y1, x2, y2 = map(int, racket["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            frame,
            "Racket",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    if tracked_ball is not None:
        x1, y1, x2, y2 = map(int, tracked_ball["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Ball",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        cx, cy = tracked_ball["center"]
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    if event is not None:
        text = f"{event['shot_type']} | {event['timestamp']:.2f}s"
        cv2.putText(
            frame,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

    return frame


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = PadelObjectDetector(
        weights=WEIGHTS_PATH,
        device="cuda:0",
        imgsz=IMGSZ,
    )
    ball_tracker = BallTracker()
    shot_classifier = ShotClassifier()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    prev_frame = None
    shot_events = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect_frame(
            frame=frame,
            prev_frame=prev_frame,
            conf_threshold=CONF,
            iou_threshold=IOU,
        )

        tracked_ball = ball_tracker.update(detections["ball_candidates"])

        event = shot_classifier.classify(
            frame_idx=frame_idx,
            fps=fps,
            players=detections["players"],
            ball_history=ball_tracker.history,
        )

        if event is not None:
            shot_events.append(event)

        annotated = draw_detections(
            frame.copy(),
            players=detections["players"],
            rackets=detections["rackets"],
            tracked_ball=tracked_ball,
            event=event,
        )

        writer.write(annotated)

        display = resize_for_display(annotated, DISPLAY_WIDTH)
        cv2.imshow("Padel Analytics", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        prev_frame = frame.copy()
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(shot_events, f, indent=2)

    pd.DataFrame(shot_events).to_csv(OUTPUT_CSV, index=False)

    summary = {
        "total_shots": len(shot_events),
        "forehand": sum(1 for x in shot_events if x["shot_type"] == "forehand"),
        "backhand": sum(1 for x in shot_events if x["shot_type"] == "backhand"),
        "serve_or_smash": sum(1 for x in shot_events if x["shot_type"] == "serve_or_smash"),
    }

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved video: {OUTPUT_VIDEO}")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
