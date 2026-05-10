from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def bbox_center(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


class BallTracker:
    def __init__(
        self,
        max_history: int = 32,
        max_missed: int = 10,
        max_distance: float = 300.0,
    ) -> None:
        self.max_history = max_history
        self.max_missed = max_missed
        self.max_distance = max_distance

        self.history: List[Tuple[float, float]] = []
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.missed_frames = 0
        self.current_ball: Optional[dict] = None

    def update(self, candidates: List[dict]) -> Optional[dict]:
        predicted_center = self._predict_center()

        if not candidates:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed:
                self.current_ball = None
            return self.current_ball

        best_candidate = self._select_best_candidate(candidates, predicted_center)

        if best_candidate is None:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed:
                self.current_ball = None
            return self.current_ball

        center = best_candidate["center"]

        if self.history:
            self.velocity = np.array(center, dtype=np.float32) - np.array(
                self.history[-1], dtype=np.float32
            )

        self.history.append(center)
        self.history = self.history[-self.max_history :]
        self.current_ball = best_candidate
        self.missed_frames = 0
        return self.current_ball

    def _predict_center(self) -> Optional[Tuple[float, float]]:
        if not self.history:
            return None
        if len(self.history) == 1:
            return self.history[-1]

        last = np.array(self.history[-1], dtype=np.float32)
        predicted = last + self.velocity
        return float(predicted[0]), float(predicted[1])

    def _select_best_candidate(
        self,
        candidates: List[dict],
        predicted_center: Optional[Tuple[float, float]],
    ) -> Optional[dict]:
        if predicted_center is None:
            return max(candidates, key=lambda c: float(c["confidence"]))

        pred = np.array(predicted_center, dtype=np.float32)
        last = np.array(self.history[-1], dtype=np.float32) if self.history else pred

        scored = []

        for candidate in candidates:
            center = np.array(candidate["center"], dtype=np.float32)

            distance_to_prediction = float(np.linalg.norm(center - pred))
            if distance_to_prediction > self.max_distance:
                continue

            distance_score = max(
                0.0, 1.0 - distance_to_prediction / self.max_distance
            )

            speed = float(np.linalg.norm(center - last))
            speed_score = self._speed_score(speed)

            final_score = (
                0.45 * float(candidate["confidence"])
                + 0.35 * distance_score
                + 0.20 * speed_score
            )

            scored.append((final_score, candidate))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        return max(candidates, key=lambda c: float(c["confidence"]))

    def _speed_score(self, speed: float) -> float:
        min_useful = 8.0
        ideal = 40.0
        max_useful = self.max_distance

        if speed < min_useful:
            return 0.0
        if speed <= ideal:
            return (speed - min_useful) / max(ideal - min_useful, 1e-6)
        if speed >= max_useful:
            return 0.0

        return max(0.0, 1.0 - (speed - ideal) / max(max_useful - ideal, 1e-6))
