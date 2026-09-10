#!/usr/bin/env python3
"""
Hand Gesture & Motion Detection
================================
Real-time hand gesture recognition + motion (swipe) detection
using MediaPipe Gesture Recognizer and OpenCV.
"""

from __future__ import annotations

import argparse
import collections
import math
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"

# Colors (BGR)
COLOR_LANDMARK = (0, 255, 0)
COLOR_CONNECTION = (255, 200, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (30, 30, 30)
COLOR_SWIPE = (0, 165, 255)
COLOR_TRAIL = (0, 255, 255)
COLOR_ACCENT = (0, 200, 255)

# Gesture emoji mapping for nicer display
GESTURE_EMOJI = {
    "None": "",
    "Closed_Fist": "✊",
    "Open_Palm": "✋",
    "Pointing_Up": "☝️",
    "Thumb_Down": "👎",
    "Thumb_Up": "👍",
    "Victory": "✌️",
    "ILoveYou": "🤟",
}

# Hand landmark connections (MediaPipe standard)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (5, 9), (9, 13), (13, 17),               # Palm
]


class MotionTracker:
    """Tracks palm center history and detects simple swipe gestures."""

    def __init__(
        self,
        history_len: int = 12,
        swipe_threshold: float = 0.12,   # fraction of frame width/height
        min_velocity: float = 0.015,
        cooldown_frames: int = 18,
    ):
        self.history: Deque[Tuple[float, float]] = collections.deque(maxlen=history_len)
        self.swipe_threshold = swipe_threshold
        self.min_velocity = min_velocity
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0
        self.last_swipe: Optional[str] = None
        self.last_swipe_time = 0.0

    def update(self, palm_x: float, palm_y: float) -> Optional[str]:
        """Add a new palm position (normalized 0-1) and return swipe name if detected."""
        self.history.append((palm_x, palm_y))

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if len(self.history) < 6:
            return None

        # Use older vs newer points for direction
        older = self.history[0]
        newer = self.history[-1]
        dx = newer[0] - older[0]
        dy = newer[1] - older[1]

        dist = math.hypot(dx, dy)
        if dist < self.min_velocity * len(self.history):
            return None

        abs_dx, abs_dy = abs(dx), abs(dy)

        swipe = None
        if abs_dx > abs_dy and abs_dx > self.swipe_threshold:
            swipe = "Swipe Right" if dx > 0 else "Swipe Left"
        elif abs_dy > abs_dx and abs_dy > self.swipe_threshold:
            swipe = "Swipe Down" if dy > 0 else "Swipe Up"

        if swipe:
            self.cooldown = self.cooldown_frames
            self.last_swipe = swipe
            self.last_swipe_time = time.time()
            # Clear history so next swipe needs a fresh motion
            self.history.clear()
            return swipe
        return None

    def get_trail_points(self, frame_w: int, frame_h: int) -> List[Tuple[int, int]]:
        return [(int(x * frame_w), int(y * frame_h)) for x, y in self.history]

    def clear(self):
        self.history.clear()
        self.last_swipe = None


class HandGestureApp:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        camera_id: int = 0,
        video_path: Optional[str] = None,
        output_path: Optional[str] = None,
        max_hands: int = 2,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download it from:\n"
                "https://storage.googleapis.com/mediapipe-models/"
                "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            )

        self.camera_id = camera_id
        self.video_path = video_path
        self.output_path = output_path
        self.show_trail = True
        self.show_motion_label = True

        # One tracker per hand index
        self.trackers: Dict[int, MotionTracker] = {
            i: MotionTracker() for i in range(max_hands)
        }

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_idx = 0
        self.fps = 0.0
        self._prev_time = time.time()

    def _open_capture(self) -> cv2.VideoCapture:
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
        else:
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.camera_id}")
            # Prefer a reasonable resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _init_writer(self, frame: np.ndarray):
        if self.output_path and self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, 25.0, (w, h))

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: List,
    ):
        h, w = frame.shape[:2]
        points = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 4, COLOR_LANDMARK, -1, cv2.LINE_AA)

        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], COLOR_CONNECTION, 2, cv2.LINE_AA)

    def _palm_center(self, landmarks) -> Tuple[float, float]:
        """Approximate palm center using wrist + middle MCP + ring MCP + index MCP."""
        # Indices: 0=wrist, 5=index_mcp, 9=middle_mcp, 13=ring_mcp, 17=pinky_mcp
        idxs = [0, 5, 9, 13, 17]
        xs = [landmarks[i].x for i in idxs]
        ys = [landmarks[i].y for i in idxs]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _draw_hud(
        self,
        frame: np.ndarray,
        gesture_name: str,
        score: float,
        hand_idx: int,
        swipe: Optional[str],
    ):
        h, w = frame.shape[:2]
        emoji = GESTURE_EMOJI.get(gesture_name, "")
        label = f"{emoji} {gesture_name}" if emoji else gesture_name
        conf = f"{score * 100:.0f}%"

        # Semi-transparent panel
        panel_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10 + hand_idx * (panel_h + 10)),
                      (340, 10 + hand_idx * (panel_h + 10) + panel_h),
                      COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        y0 = 40 + hand_idx * (panel_h + 10)
        cv2.putText(frame, f"Hand {hand_idx + 1}", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (20, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(frame, conf, (20, y0 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

        if swipe and self.show_motion_label:
            # Big centered swipe notification
            text = swipe
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cx, cy = w // 2, h // 2 - 40
            cv2.rectangle(frame, (cx - tw // 2 - 20, cy - th - 15),
                          (cx + tw // 2 + 20, cy + 15), (0, 0, 0), -1)
            cv2.putText(frame, text, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLOR_SWIPE, 3, cv2.LINE_AA)

    def _draw_fps(self, frame: np.ndarray):
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 140, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        h, w = frame.shape[:2]
        current_swipes: List[Optional[str]] = [None] * len(self.trackers)

        if result.hand_landmarks:
            for i, (landmarks, gestures) in enumerate(
                zip(result.hand_landmarks, result.gestures)
            ):
                if i >= len(self.trackers):
                    break

                # Draw skeleton
                self._draw_landmarks(frame, landmarks)

                # Gesture
                gesture_name = "None"
                score = 0.0
                if gestures:
                    top = gestures[0]
                    gesture_name = top.category_name
                    score = top.score

                # Motion
                palm_x, palm_y = self._palm_center(landmarks)
                swipe = self.trackers[i].update(palm_x, palm_y)
                current_swipes[i] = swipe

                # Trail
                if self.show_trail:
                    pts = self.trackers[i].get_trail_points(w, h)
                    for j in range(1, len(pts)):
                        alpha = j / len(pts)
                        thickness = max(1, int(3 * alpha))
                        color = (
                            int(COLOR_TRAIL[0] * alpha),
                            int(COLOR_TRAIL[1] * alpha),
                            int(COLOR_TRAIL[2] * alpha),
                        )
                        cv2.line(frame, pts[j - 1], pts[j], color, thickness, cv2.LINE_AA)
                    if pts:
                        cv2.circle(frame, pts[-1], 6, COLOR_TRAIL, -1, cv2.LINE_AA)

                # HUD for this hand
                # Keep showing last swipe for a short time
                display_swipe = swipe
                if not display_swipe and self.trackers[i].last_swipe:
                    if time.time() - self.trackers[i].last_swipe_time < 1.2:
                        display_swipe = self.trackers[i].last_swipe

                self._draw_hud(frame, gesture_name, score, i, display_swipe)

        self._draw_fps(frame)

        # Help text
        cv2.putText(frame, "q:quit  s:trail  c:clear  m:motion", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        return frame

    def run(self):
        cap = self._open_capture()
        print("▶ Hand Gesture & Motion Detection started")
        print("  Press 'q' to quit\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror for more natural interaction when using webcam
                if not self.video_path:
                    frame = cv2.flip(frame, 1)

                self._init_writer(frame)

                timestamp_ms = int(self.frame_idx * (1000 / 30))  # assume ~30 fps
                self.frame_idx += 1

                annotated = self.process_frame(frame, timestamp_ms)

                if self.writer:
                    self.writer.write(annotated)

                cv2.imshow("Hand Gesture & Motion Detection", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self.show_trail = not self.show_trail
                    print(f"  Trail: {'ON' if self.show_trail else 'OFF'}")
                elif key == ord("c"):
                    for t in self.trackers.values():
                        t.clear()
                    print("  Trails cleared")
                elif key == ord("m"):
                    self.show_motion_label = not self.show_motion_label
                    print(f"  Motion labels: {'ON' if self.show_motion_label else 'OFF'}")

        finally:
            cap.release()
            if self.writer:
                self.writer.release()
            self.recognizer.close()
            cv2.destroyAllWindows()
            print("■ Stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Hand Gesture & Motion Detection")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--video", type=str, default=None, help="Path to input video file")
    p.add_argument("--output", type=str, default=None, help="Path to save output video")
    p.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to .task model")
    return p.parse_args()


def main():
    args = parse_args()
    app = HandGestureApp(
        model_path=Path(args.model),
        camera_id=args.camera,
        video_path=args.video,
        output_path=args.output,
    )
    app.run()


if __name__ == "__main__":
    main()
