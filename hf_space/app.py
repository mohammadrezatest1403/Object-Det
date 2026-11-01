import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(page_title="Object Detection", page_icon="🎯", layout="wide")

st.title("🎯 Object Detection with YOLOv8")
st.markdown("Upload an image and YOLOv8 will detect and localize objects in it.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

with st.sidebar:
    st.header("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    st.caption("Model: YOLOv8n (COCO, 80 classes)")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original", use_container_width=True)

    with st.spinner("Detecting..."):
        t = time.time()
        results = model(img, conf=conf)
        inference_ms = (time.time() - t) * 1000
        annotated = results[0].plot()[:, :, ::-1]

    with col2:
        st.image(annotated, caption="Detected", use_container_width=True)

    st.caption(f"⏱️ Inference: {inference_ms:.1f} ms")

    st.subheader("Detections")
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        data = []
        for i, box in enumerate(boxes):
            cls_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            data.append({"#": i + 1, "Class": cls_name, "Confidence": f"{confidence:.1%}"})
        st.table(data)
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")