#!/usr/bin/env python3
"""
Hand Gesture & Motion Detection - Real-time Web Version
Uses Streamlit + streamlit-webrtc for continuous live camera streaming.
Deployable on Render / Streamlit Cloud.
"""

from __future__ import annotations

import collections
import math
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"

GESTURE_EMOJI = {
    "None": "",
    "Closed_Fist": "Fist",
    "Open_Palm": "Open Palm",
    "Pointing_Up": "Pointing Up",
    "Thumb_Down": "Thumb Down",
    "Thumb_Up": "Thumb Up",
    "Victory": "Victory",
    "ILoveYou": "I Love You",
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

COLOR_LANDMARK = (0, 255, 0)
COLOR_CONNECTION = (255, 200, 100)
COLOR_TRAIL = (0, 255, 255)
COLOR_SWIPE = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (40, 40, 40)
COLOR_ACCENT = (0, 200, 255)


# ---------------------------------------------------------------------------
# Motion Tracker
# ---------------------------------------------------------------------------
class MotionTracker:
    def __init__(
        self,
        history_len: int = 10,
        swipe_threshold: float = 0.13,
        min_velocity: float = 0.012,
        cooldown_frames: int = 15,
    ):
        self.history: Deque[Tuple[float, float]] = collections.deque(maxlen=history_len)
        self.swipe_threshold = swipe_threshold
        self.min_velocity = min_velocity
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0
        self.last_swipe: Optional[str] = None
        self.last_swipe_time = 0.0

    def update(self, palm_x: float, palm_y: float) -> Optional[str]:
        self.history.append((palm_x, palm_y))

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if len(self.history) < 5:
            return None

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
            self.history.clear()
            return swipe
        return None

    def get_trail(self, w: int, h: int) -> List[Tuple[int, int]]:
        return [(int(x * w), int(y * h)) for x, y in self.history]

    def clear(self):
        self.history.clear()
        self.last_swipe = None


# ---------------------------------------------------------------------------
# Video Processor (runs on every frame)
# ---------------------------------------------------------------------------
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.recognizer = None
        self.trackers: Dict[int, MotionTracker] = {0: MotionTracker(), 1: MotionTracker()}
        self.frame_count = 0
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def _draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        h, w = frame.shape[:2]
        points = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 4, COLOR_LANDMARK, -1, cv2.LINE_AA)

        for s, e in HAND_CONNECTIONS:
            if s < len(points) and e < len(points):
                cv2.line(frame, points[s], points[e], COLOR_CONNECTION, 2, cv2.LINE_AA)

    def _palm_center(self, landmarks) -> Tuple[float, float]:
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
    ) -> None:
        h, w = frame.shape[:2]
        label = GESTURE_EMOJI.get(gesture_name, gesture_name) or gesture_name
        conf = f"{score * 100:.0f}%"

        panel_h = 70
        y_offset = 10 + hand_idx * (panel_h + 8)

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, y_offset), (300, y_offset + panel_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, f"Hand {hand_idx + 1}", (18, y_offset + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, label, (18, y_offset + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_ACCENT, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, conf, (200, y_offset + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA
        )

        # Show swipe for a short time
        if swipe or (self.trackers[hand_idx].last_swipe and
                     time.time() - self.trackers[hand_idx].last_swipe_time < 1.1):
            text = swipe or self.trackers[hand_idx].last_swipe
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
            cx, cy = w // 2, h // 2 - 30
            cv2.rectangle(
                frame,
                (cx - tw // 2 - 18, cy - th - 12),
                (cx + tw // 2 + 18, cy + 12),
                (0, 0, 0), -1
            )
            cv2.putText(
                frame, text, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLOR_SWIPE, 3, cv2.LINE_AA
            )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror for natural feel

        h, w = img.shape[:2]
        self.frame_count += 1
        timestamp_ms = self.frame_count * 33  # ~30 fps

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for i, (landmarks, gestures) in enumerate(
                zip(result.hand_landmarks, result.gestures)
            ):
                if i > 1:
                    break

                self._draw_landmarks(img, landmarks)

                gesture_name = "None"
                score = 0.0
                if gestures:
                    top = gestures[0]
                    gesture_name = top.category_name
                    score = top.score

                palm_x, palm_y = self._palm_center(landmarks)
                swipe = self.trackers[i].update(palm_x, palm_y)

                # Draw trail
                pts = self.trackers[i].get_trail(w, h)
                for j in range(1, len(pts)):
                    alpha = j / max(len(pts), 1)
                    thickness = max(1, int(3 * alpha))
                    color = (
                        int(COLOR_TRAIL[0] * alpha),
                        int(COLOR_TRAIL[1] * alpha),
                        int(COLOR_TRAIL[2] * alpha),
                    )
                    cv2.line(img, pts[j - 1], pts[j], color, thickness, cv2.LINE_AA)
                if pts:
                    cv2.circle(img, pts[-1], 6, COLOR_TRAIL, -1, cv2.LINE_AA)

                self._draw_hud(img, gesture_name, score, i, swipe)

        # Small help text
        cv2.putText(
            img, "Live Hand Gesture + Motion Detection",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Hand Gesture Detection - Live",
        page_icon="✋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("✋ Real-time Hand Gesture & Motion Detection")
    st.caption("Live camera streaming with MediaPipe Gesture Recognizer + swipe detection")

    with st.sidebar:
        st.header("How to use")
        st.markdown("""
1. Click **START** below the video  
2. Allow camera access in the browser  
3. Show your hand to the camera  
4. Try different gestures and swipes
        """)
        st.markdown("---")
        st.markdown("### Supported Gestures")
        st.markdown("""
- 👍 **Thumb Up**
- 👎 **Thumb Down**
- ✌️ **Victory** (Peace)
- ☝️ **Pointing Up**
- ✊ **Closed Fist**
- ✋ **Open Palm**
- 🤟 **I Love You**
        """)
        st.markdown("---")
        st.markdown("### Motion Detection")
        st.markdown("Swipe **Left / Right / Up / Down** with your hand.")
        st.markdown("---")
        st.info("Works best with good lighting and a clear background.")

    st.markdown("### Live Camera")
    st.write("Click the **START** button, then allow camera permission.")

    webrtc_streamer(
        key="hand-gesture",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=GestureProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("---")
    st.markdown(
        "Powered by **MediaPipe** + **Streamlit** + **streamlit-webrtc**"
    )


if __name__ == "__main__":
    main()
