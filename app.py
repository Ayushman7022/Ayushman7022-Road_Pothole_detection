import streamlit as st
import cv2
import torch
import time
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
import os

# ------------------------------------------------------------
# 🎨 Streamlit App Config
# ------------------------------------------------------------
st.set_page_config(page_title="🚗 Road Damage Detection", layout="wide")
st.markdown("<h1 style='text-align:center;color:#ff4b4b;'>🚗 Real-Time Road Damage Detection </h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar with app info
st.sidebar.title("⚙️ App Settings")
st.sidebar.markdown("Fine-tune detection behavior below 👇")

# ------------------------------------------------------------
# 🧠 Sidebar Settings
# ------------------------------------------------------------
conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
cooldown_time = 10  # seconds between screenshots of same class
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"🖥️ Using device: **{device.upper()}**")

# ------------------------------------------------------------
# 🌍 Static Location (via Geocode API)
# ------------------------------------------------------------
GEOCODE_URL = "https://geocode.maps.co/search"
API_KEY = "6910ea65768a7691320498ytfd0eca0"

@st.cache_data
def get_lat_lon(address="Jalandhar"):
    try:
        resp = requests.get(GEOCODE_URL, params={"q": address, "api_key": API_KEY}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, None

        best = next(
            (d for d in data if d.get("class") == "place" and d.get("type") in ("city", "town")),
            data[0]
        )
        return float(best["lat"]), float(best["lon"])
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch location: {e}")
        return None, None

lat, lon = get_lat_lon("Jalandhar")
if lat and lon:
    st.sidebar.info(f"📍 **Location:** Jalandhar\n\nLat: `{lat:.6f}`\nLon: `{lon:.6f}`")
else:
    st.sidebar.error("Could not fetch coordinates ❌")

# ------------------------------------------------------------
# 🧠 Load YOLOv8 Model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to(device)
    return model

model = load_model()

# ------------------------------------------------------------
# 🖼️ UI Layout
# ------------------------------------------------------------
left, right = st.columns([1.6, 1])
left.subheader("🎥 Live Camera Feed")
right.subheader("📸 Detected Detections")

frame_placeholder = left.empty()
table_placeholder = right.empty()

# Prepare folder for detections
os.makedirs("detections", exist_ok=True)
detection_records = []
last_screenshot_time = {}
screenshot_count = 0

# ------------------------------------------------------------
# 📷 Webcam Capture
# ------------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Cannot access the webcam.")
else:
    st.success("✅ Webcam connected! Click below to start detection.")
    stop_button = st.button("🛑 Stop Stream")

    with st.sidebar:
        screenshot_display = st.empty()

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Frame not captured.")
            break

        frame = cv2.resize(frame, (640, 480))
        results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = model.names[int(box.cls)]
                label = f"{cls} ({conf:.2f})"

                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Save screenshot if confidence > threshold & cooldown passed
                current_time = time.time()
                last_time = last_screenshot_time.get(cls, 0)
                if conf >= conf_threshold and (current_time - last_time) > cooldown_time:
                    timestamp = int(current_time)
                    filename = f"detections/{cls}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)

                    detection_records.append({
                        "Class": cls,
                        "Confidence": f"{conf:.2f}",
                        "Latitude": f"{lat:.6f}",
                        "Longitude": f"{lon:.6f}",
                        "Image": filename,
                        "Timestamp": time.strftime("%H:%M:%S", time.localtime(timestamp))
                    })

                    last_screenshot_time[cls] = current_time
                    screenshot_count += 1
                    screenshot_display.info(f"📸 **Screenshots Captured:** {screenshot_count}")

        # Display live frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Show detections in right column
        if detection_records:
            latest_records = detection_records[-5:][::-1]
            table_placeholder.dataframe(latest_records, use_container_width=True)

        time.sleep(0.03)

    cap.release()
    st.warning("🛑 Stream stopped.")
