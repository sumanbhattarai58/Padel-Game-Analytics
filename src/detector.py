from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


Detection = Dict[str, object]


class PadelObjectDetector:
    PLAYER_CLASS_ID = 1
    RACKET_CLASS_ID = 2

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        imgsz: int = 768,
    ) -> None:
        self.model = YOLO(weights)
        self.model.to(device)

        self.device = device
        self.imgsz = imgsz
        self.yolo_classes = [self.PLAYER_CLASS_ID, self.RACKET_CLASS_ID]
        self.ball_detector = BallCandidateDetector()

    def detect_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
    ) -> Dict[str, List[Detection]]:
        result = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.yolo_classes,
            verbose=False,
        )[0]

        parsed = self._parse_yolo_result(result)
        parsed["ball_candidates"] = self.ball_detector.detect_candidates(
            frame=frame,
            prev_frame=prev_frame,
            players=parsed["players"],
            rackets=parsed["rackets"],
        )
        return parsed

    def _parse_yolo_result(self, result) -> Dict[str, List[Detection]]:
        players: List[Detection] = []
        rackets: List[Detection] = []

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return {"players": players, "rackets": rackets}

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = box.xyxy[0].cpu().numpy().astype(np.float32)

            det: Detection = {
                "bbox": bbox,
                "confidence": confidence,
                "class_id": class_id,
                "source": "yolo",
            }

            if class_id == self.PLAYER_CLASS_ID:
                players.append(det)
            elif class_id == self.RACKET_CLASS_ID:
                rackets.append(det)

        return {"players": players, "rackets": rackets}


class BallCandidateDetector:
    def __init__(
        self,
        min_radius: int = 2,
        max_radius: int = 12,
        min_area: int = 4,
        max_area: int = 220,
        diff_threshold: int = 8,
        white_threshold: int = 170,
        min_circularity: float = 0.08,
        ignore_top_ratio: float = 0.12,
        ignore_left_ratio: float = 0.08,
        max_candidates: int = 12,
    ) -> None:
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_area = min_area
        self.max_area = max_area
        self.diff_threshold = diff_threshold
        self.white_threshold = white_threshold
        self.min_circularity = min_circularity
        self.ignore_top_ratio = ignore_top_ratio
        self.ignore_left_ratio = ignore_left_ratio
        self.max_candidates = max_candidates

    def detect_candidates(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        players: Optional[List[Detection]] = None,
        rackets: Optional[List[Detection]] = None,
    ) -> List[Detection]:
        if prev_frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)

        diff = cv2.absdiff(gray_blur, prev_blur)
        _, motion_mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        _, white_mask = cv2.threshold(gray_blur, self.white_threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_and(motion_mask, white_mask)

        h, w = gray.shape
        mask[: int(h * self.ignore_top_ratio), :] = 0
        mask[:, : int(w * self.ignore_left_ratio)] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
            if circularity < self.min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            center = (float(x), float(y))
            if self._inside_any_box(center, players):
                continue
            if self._inside_any_box(center, rackets):
                continue

            x1 = max(0, int(x - radius))
            y1 = max(0, int(y - radius))
            x2 = min(w - 1, int(x + radius))
            y2 = min(h - 1, int(y + radius))

            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            roi_gray = gray[y1:y2, x1:x2]
            roi_motion = motion_mask[y1:y2, x1:x2]
            if roi_gray.size == 0 or roi_motion.size == 0:
                continue

            brightness = float(np.mean(roi_gray)) / 255.0
            motion_strength = float(np.mean(roi_motion)) / 255.0

            size_score = self._size_score(radius)

            score = (
                0.45 * motion_strength
                + 0.15 * brightness
                + 0.10 * min(circularity, 1.0)
                + 0.10 * size_score
            )

            candidates.append(
                {
                    "bbox": bbox,
                    "center": center,
                    "confidence": float(score),
                    "radius": float(radius),
                    "area": float(area),
                    "circularity": float(circularity),
                    "motion_strength": float(motion_strength),
                    "brightness": float(brightness),
                    "size_score": float(size_score),
                    "source": "motion_candidate",
                }
            )

        candidates.sort(key=lambda det: float(det["confidence"]), reverse=True)
        return candidates[: self.max_candidates]

    def _inside_any_box(
        self,
        center: tuple[float, float],
        detections: Optional[List[Detection]],
    ) -> bool:
        if not detections:
            return False

        cx, cy = center
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def _size_score(self, radius: float) -> float:
        ideal = 6.0
        tolerance = 6.0
        return max(0.0, 1.0 - abs(radius - ideal) / tolerance)
