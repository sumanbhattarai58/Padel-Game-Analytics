from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from tracker import bbox_center


def bbox_height(bbox) -> float:
    return max(1.0, bbox[3] - bbox[1])


def direction_change_deg(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> float:
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0

    cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class ShotClassifier:
    def __init__(
        self,
        contact_distance_threshold: float = 150.0,
        trajectory_change_threshold: float = 30.0,
        smash_height_ratio: float = 0.40,
        cooldown_frames: int = 8,
    ) -> None:
        self.contact_distance_threshold = contact_distance_threshold
        self.trajectory_change_threshold = trajectory_change_threshold
        self.smash_height_ratio = smash_height_ratio
        self.cooldown_frames = cooldown_frames
        self.last_event_frame = -1000

    def classify(
        self,
        frame_idx: int,
        fps: float,
        players: List[dict],
        ball_history: List[Tuple[float, float]],
    ) -> Optional[dict]:
        if len(ball_history) < 4:
            return None

        if frame_idx - self.last_event_frame < self.cooldown_frames:
            return None

        ball_center = ball_history[-1]
        hitter = self._nearest_player(players, ball_center)
        if hitter is None:
            return None

        distance = np.linalg.norm(np.array(bbox_center(hitter["bbox"])) - np.array(ball_center))
        if distance > self.contact_distance_threshold:
            return None

        angle = direction_change_deg(ball_history[-4], ball_history[-2], ball_history[-1])
        if angle < self.trajectory_change_threshold:
            return None

        shot_type = self._classify_shot_type(hitter["bbox"], ball_center)
        self.last_event_frame = frame_idx

        return {
            "frame": frame_idx,
            "timestamp": round(frame_idx / fps, 3),
            "player_id": None,
            "shot_type": shot_type,
            "confidence": round(min(0.95, 0.55 + angle / 180.0), 3),
        }

    def _nearest_player(
        self,
        players: List[dict],
        ball_center: Tuple[float, float],
    ) -> Optional[dict]:
        if not players:
            return None

        best_player = None
        best_distance = float("inf")

        for player in players:
            center = bbox_center(player["bbox"])
            distance = np.linalg.norm(np.array(center) - np.array(ball_center))
            if distance < best_distance:
                best_distance = distance
                best_player = player

        return best_player

    def _classify_shot_type(
        self,
        player_bbox,
        ball_center: Tuple[float, float],
    ) -> str:
        x1, y1, x2, y2 = player_bbox
        center_x = (x1 + x2) * 0.5
        height = bbox_height(player_bbox)

        relative_x = ball_center[0] - center_x
        relative_y_ratio = (ball_center[1] - y1) / height

        if relative_y_ratio <= self.smash_height_ratio:
            return "serve_or_smash"
        if relative_x >= 0:
            return "forehand"
        return "backhand"
