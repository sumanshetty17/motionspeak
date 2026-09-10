#!/usr/bin/env python3
"""
Hand Gesture & Motion Detection - Web Version (Streamlit)
Deployable on Render / Streamlit Cloud / Hugging Face Spaces
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"

GESTURE_EMOJI = {
    "None": "❓",
    "Closed_Fist": "✊",
    "Open_Palm": "✋",
    "Pointing_Up": "☝️",
    "Thumb_Down": "👎",
    "Thumb_Up": "👍",
    "Victory": "✌️",
    "ILoveYou": "🤟",
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


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_recognizer():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def draw_landmarks(image: np.ndarray, landmarks) -> np.ndarray:
    h, w = image.shape[:2]
    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(image, (x, y), 5, COLOR_LANDMARK, -1, cv2.LINE_AA)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(image, points[start], points[end], COLOR_CONNECTION, 2, cv2.LINE_AA)
    return image


def palm_center(landmarks) -> Tuple[float, float]:
    idxs = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x for i in idxs]
    ys = [landmarks[i].y for i in idxs]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def process_image(image_bgr: np.ndarray, recognizer) -> Tuple[np.ndarray, List[dict]]:
    """Run gesture recognition and return annotated image + results."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = recognizer.recognize(mp_image)

    annotated = image_bgr.copy()
    detections = []

    if result.hand_landmarks:
        for i, (landmarks, gestures) in enumerate(
            zip(result.hand_landmarks, result.gestures)
        ):
            annotated = draw_landmarks(annotated, landmarks)

            gesture_name = "None"
            score = 0.0
            if gestures:
                top = gestures[0]
                gesture_name = top.category_name
                score = top.score

            px, py = palm_center(landmarks)
            detections.append(
                {
                    "hand": i + 1,
                    "gesture": gesture_name,
                    "score": score,
                    "palm": (px, py),
                    "emoji": GESTURE_EMOJI.get(gesture_name, ""),
                }
            )

    return annotated, detections


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Hand Gesture Detection",
        page_icon="✋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("✋ Hand Gesture & Motion Detection")
    st.caption("Real-time hand gesture recognition powered by MediaPipe")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Input Mode",
            ["📷 Webcam (Camera)", "🖼️ Upload Image", "🎬 Upload Video"],
            index=0,
        )
        st.markdown("---")
        st.markdown("### Supported Gestures")
        for name, emoji in GESTURE_EMOJI.items():
            if name != "None":
                st.write(f"{emoji}  **{name}**")
        st.markdown("---")
        st.markdown(
            "Made with [MediaPipe](https://developers.google.com/mediapipe) + Streamlit"
        )

    recognizer = load_recognizer()

    # ---------------- Webcam mode ----------------
    if mode == "📷 Webcam (Camera)":
        st.info("Click **Take Photo** below. Allow camera access when the browser asks.")
        camera_img = st.camera_input("Capture your hand gesture", label_visibility="collapsed")

        if camera_img is not None:
            # Convert to OpenCV
            pil_img = Image.open(camera_img)
            img_np = np.array(pil_img)
            # camera_input gives RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Detecting gestures..."):
                annotated, detections = process_image(img_bgr, recognizer)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(pil_img, use_container_width=True)
            with col2:
                st.subheader("Detected")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            if detections:
                st.subheader("Results")
                for d in detections:
                    conf = d["score"] * 100
                    st.success(
                        f"**Hand {d['hand']}**: {d['emoji']} **{d['gesture']}** "
                        f"(confidence: {conf:.1f}%)"
                    )
            else:
                st.warning("No hand detected. Try better lighting or show your hand clearly.")

    # ---------------- Image upload ----------------
    elif mode == "🖼️ Upload Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Detecting gestures..."):
                annotated, detections = process_image(img_bgr, recognizer)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(pil_img, use_container_width=True)
            with col2:
                st.subheader("Detected")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            if detections:
                st.subheader("Results")
                for d in detections:
                    conf = d["score"] * 100
                    st.success(
                        f"**Hand {d['hand']}**: {d['emoji']} **{d['gesture']}** "
                        f"(confidence: {conf:.1f}%)"
                    )
            else:
                st.warning("No hand detected.")

    # ---------------- Video upload ----------------
    else:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            # Save temp video
            tfile = Path("temp_video.mp4")
            tfile.write_bytes(uploaded_video.read())

            st.info("Processing video (this may take a moment)...")
            cap = cv2.VideoCapture(str(tfile))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            progress = st.progress(0)
            status = st.empty()
            result_placeholder = st.empty()

            frame_count = 0
            sample_every = max(1, int(fps // 5))  # process ~5 fps
            last_detections = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % sample_every != 0:
                    continue

                annotated, detections = process_image(frame, recognizer)
                last_detections = detections

                # Show current frame
                result_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_count}/{total_frames}",
                    use_container_width=True,
                )

                progress.progress(min(frame_count / total_frames, 1.0))
                if detections:
                    texts = [
                        f"{d['emoji']} {d['gesture']} ({d['score']*100:.0f}%)"
                        for d in detections
                    ]
                    status.success(" | ".join(texts))
                else:
                    status.info("No hand detected in this frame")

            cap.release()
            tfile.unlink(missing_ok=True)
            progress.empty()
            st.success("Video processing finished!")

            if last_detections:
                st.subheader("Last detected gestures")
                for d in last_detections:
                    st.write(f"{d['emoji']} **{d['gesture']}** – {d['score']*100:.1f}%")


if __name__ == "__main__":
    main()
