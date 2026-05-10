from ultralytics import YOLO


MODEL_NAME = "yolov8s.pt"
DATA_YAML = "data\dataset\data.yaml"


def main() -> None:
    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=20,
        imgsz=640,
        batch=2,
        device=0,
        workers=2,
        project="models",
        name="padel_detector",
    )


if __name__ == "__main__":
    main()
