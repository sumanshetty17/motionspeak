# Hand Gesture & Motion Detection (Real-time)

Live real-time hand gesture recognition + swipe detection using **MediaPipe** + **Streamlit** + **WebRTC**.

## Features

- Continuous live camera stream in the browser
- 7 gestures: Thumb Up, Thumb Down, Victory, Pointing Up, Closed Fist, Open Palm, I Love You
- Hand landmarks + skeleton overlay
- Motion trail + Swipe detection (Left / Right / Up / Down)

## Deploy on Render (Recommended – Docker)

This is the **most reliable** way and fixes the `libGLESv2` error.

### Steps:

1. Push this entire folder to a **GitHub repository**.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
3. Connect your GitHub repo.
4. Important settings:
   - **Runtime**: **Docker**
   - **Dockerfile Path**: `./Dockerfile` (default is fine)
5. Click **Create Web Service**.

Render will build the Docker image (takes a few minutes the first time) and give you a public URL.

Then open the URL → click **START** → allow camera access.

> Free tier on Render may sleep after inactivity. Just refresh the page to wake it up.

## Local Run (without Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Local Run with Docker

```bash
docker build -t hand-gesture .
docker run -p 8501:8501 hand-gesture
```

Then open http://localhost:8501

## Project Structure

```
├── app.py                      # Real-time Streamlit + WebRTC app
├── hand_gesture_detector.py    # Desktop OpenCV version (optional)
├── gesture_recognizer.task     # MediaPipe model (required)
├── Dockerfile                  # Fixes libGLESv2 on cloud
├── requirements.txt
├── render.yaml
├── Procfile
└── README.md
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `libGLESv2.so.2: cannot open shared object file` | Use **Docker** runtime on Render (this repo already has the Dockerfile) |
| Camera not starting | Allow camera permission in browser. Use HTTPS (Render provides it) |
| Blank video | Click START again or refresh the page |
| Slow performance | Free Render instances are limited. Works better on paid plans or locally |

## Tips

- Good lighting improves accuracy a lot
- Keep your hand clearly visible in the frame
- Swipe with a clear, reasonably fast movement
