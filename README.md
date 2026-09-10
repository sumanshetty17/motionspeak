# Hand Gesture & Motion Detection (Real-time Web)

**Live real-time** hand gesture recognition and motion (swipe) detection using MediaPipe + Streamlit + WebRTC.

Works with your browser camera continuously — no photo button needed.

## Features

- **Real-time live camera** streaming in the browser
- Gesture recognition (7 gestures):
  - 👍 Thumb Up
  - 👎 Thumb Down
  - ✌️ Victory
  - ☝️ Pointing Up
  - ✊ Closed Fist
  - ✋ Open Palm
  - 🤟 I Love You
- Hand landmarks + skeleton
- Motion trail
- Swipe detection (Left / Right / Up / Down)
- Multi-hand support

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL (usually http://localhost:8501), click **START**, and allow camera access.

## Deploy on Render

1. Push this folder to a GitHub repository.
2. Go to [Render](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect the repo.
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**:  
     ```
     streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
     ```
5. Deploy.

After deployment, open the Render URL → click **START** → allow camera.

> **Note**: Real-time WebRTC works best on HTTPS (Render provides HTTPS by default).

## Project Files

```
├── app.py                      # Real-time web app (main)
├── hand_gesture_detector.py    # Original desktop OpenCV version
├── gesture_recognizer.task     # MediaPipe model
├── requirements.txt
├── Procfile
└── README.md
```

## Tips

- Use good lighting
- Keep hand clearly visible
- Swipe with a reasonably clear movement
