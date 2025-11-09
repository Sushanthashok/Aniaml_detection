
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time
import os

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Animal Detector", page_icon="🐾", layout="wide")
st.title("🐾 Animal Detection & Classification")
st.caption("Detect species, highlight carnivores in red, and show a pop-up with the carnivore count. Powered by YOLOv8x.")

# -----------------------------
# Model & Settings
# -----------------------------
MODEL_PATH = "best.pt"  
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MAX_DET = 300

CARNIVORES = {
    "bear", "wolf", "fox", "lion", "tiger", "leopard", "cheetah",
    "hyena", "cougar", "lynx", "jaguar"
}

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

with st.spinner("Loading model..."):
    model = load_model()
st.success("Model ready (best.pt loaded).", icon="✅")

# -----------------------------
# Draw bounding boxes
# -----------------------------
def draw_boxes(image_bgr: np.ndarray, result) -> tuple[np.ndarray, int]:
    im = image_bgr.copy()
    names = result.names if hasattr(result, "names") else {}
    carn_count = 0

    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return im, 0

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [None] * len(cls)

    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names.get(c, str(c))
        is_carn = label.lower() in CARNIVORES
        color = (0, 0, 255) if is_carn else (0, 200, 0)
        if is_carn:
            carn_count += 1

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        text = f"{label}" + (f" {s:.2f}" if s is not None else "")
        cv2.putText(im, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return im, carn_count

def show_popup(carn_count: int, scope: str = "frame"):
    if carn_count > 0:
        st.warning(f"⚠️ {carn_count} carnivorous animal(s) detected in this {scope}!", icon="⚠️")
    else:
        st.toast(f"No carnivores detected in this {scope}.")

# -----------------------------
# UI Tabs
# -----------------------------
tab_img, tab_vid = st.tabs(["🖼️ Image", "🎬 Video"])

# ---------- IMAGE ----------
with tab_img:
    file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if file:
        bytes_data = file.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read this image. Try another file.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.spinner("Running detection..."):
                res = model.predict(img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DET, verbose=False)[0]
                drawn, carn_count = draw_boxes(img_bgr, res)

            with col2:
                st.subheader("Detections")
                st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), use_container_width=True)

            show_popup(carn_count, scope="image")

            out_bytes = cv2.imencode(".jpg", drawn)[1].tobytes()
            st.download_button("Download annotated image", data=out_bytes, file_name="detections.jpg", mime="image/jpeg")

# ---------- VIDEO ----------
with tab_vid:
    vfile = st.file_uploader("Upload a video (MP4/MOV/AVI/MKV)", type=["mp4", "mov", "avi", "mkv"])
    if vfile:
        tf_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf_in.write(vfile.read())
        tf_in.flush()

        cap = cv2.VideoCapture(tf_in.name)
        if not cap.isOpened():
            st.error("Could not open this video. Try re-encoding to H.264 MP4.")
        else:
            stframe = st.empty()
            total_carn = 0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            with st.spinner("Processing video..."):
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    res = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DET, verbose=False)[0]
                    drawn, carn_count = draw_boxes(frame, res)
                    total_carn += carn_count

                    stframe.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    writer.write(drawn)
                    frame_idx += 1

            cap.release()
            writer.release()

            show_popup(total_carn, scope="video (sum of per-frame detections)")
            with open(out_path, "rb") as f:
                st.download_button("Download annotated video", data=f.read(), file_name="detections.mp4", mime="video/mp4")
