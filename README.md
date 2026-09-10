# Hand Gesture & Motion Detection (Web Version)

Real-time / interactive hand gesture recognition using **MediaPipe Gesture Recognizer** + **Streamlit**.

Ready to deploy on **Render**, Streamlit Community Cloud, or Hugging Face Spaces.

## Features

- 📷 **Webcam capture** (browser camera)
- 🖼️ **Image upload**
- 🎬 **Video upload** (frame-by-frame processing)
- Recognizes 7 gestures:
  - 👍 Thumb_Up
  - 👎 Thumb_Down
  - ✌️ Victory
  - ☝️ Pointing_Up
  - ✊ Closed_Fist
  - ✋ Open_Palm
  - 🤟 ILoveYou
- Hand landmarks visualization
- Multi-hand support

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Deploy on Render

1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect your GitHub repo.
4. Configure:
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Click **Create Web Service**.

That’s it – Render will give you a public URL.

### Alternative: Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect the GitHub repo
3. Set main file path to `app.py`
4. Deploy

## Project Structure

```
hand_gesture_motion_detection/
├── app.py                    # Streamlit web app (main entry)
├── hand_gesture_detector.py  # Original desktop version (OpenCV window)
├── gesture_recognizer.task   # MediaPipe model
├── requirements.txt
└── README.md
```

## Notes

- The browser will ask for camera permission when you use the Webcam mode.
- Best results with good lighting and a clearly visible hand.
- The original desktop version (`hand_gesture_detector.py`) still works locally if you want continuous real-time tracking with swipe detection.
