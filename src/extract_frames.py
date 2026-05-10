from pathlib import Path
import cv2


VIDEO_PATH = Path("data/raw/input_sample_video.mp4")
OUTPUT_DIR = Path("data/frames")
SAMPLE_EVERY = 5
MAX_FRAMES = 150
RESIZE_TO = (1280, 720)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    saved = 0
    frame_idx = 0

    while saved < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % SAMPLE_EVERY == 0:
            frame = cv2.resize(frame, RESIZE_TO)
            out_path = OUTPUT_DIR / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
