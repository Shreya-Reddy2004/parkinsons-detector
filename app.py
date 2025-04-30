import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
import uuid
from predict_parkinsons import predict_image

# Paths to the models
SPIRAL_MODEL = os.path.join("models", "random_forest_spiral_model.pkl")
WAVE_MODEL = os.path.join("models", "random_forest_wave_model.pkl")

st.title("🧠 Parkinson's Detector")

st.sidebar.title("Choose Input Method")
input_mode = st.sidebar.radio("Select input type", ["Draw on Canvas", "Upload Image"])

st.sidebar.title("Brush Settings")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 12)
stroke_color = st.sidebar.color_picker("Stroke color", "#000000")

def save_temp_image(img: Image.Image) -> str:
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("temp_images", filename)
    os.makedirs("temp_images", exist_ok=True)
    img.save(filepath)
    return filepath

def run_detection(image_path: str, model_path: str):
    try:
        pred = predict_image(image_path, model_path)
        st.markdown(f"### Result: {'🟢 Healthy' if pred == 'Healthy' else '🔴 Parkinson’s'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Canvas Drawing Section
if input_mode == "Draw on Canvas":
    st.subheader("Draw a spiral or wave below")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#ffffff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img_rgba = canvas_result.image_data
        alpha = img_rgba[:, :, 3] > 0  # Alpha channel mask
        rgb_array = (img_rgba[:, :, :3] * 255).astype(np.uint8)
        white_bg = np.ones_like(rgb_array, dtype=np.uint8) * 255
        white_bg[alpha] = rgb_array[alpha]

        img = Image.fromarray(white_bg).convert("L").resize((128, 128))
        saved_path = save_temp_image(img)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌀 Detect Spiral"):
                run_detection(saved_path, SPIRAL_MODEL)

        with col2:
            if st.button("🌊 Detect Wave"):
                run_detection(saved_path, WAVE_MODEL)

# Upload Image Section
elif input_mode == "Upload Image":
    st.subheader("Upload an image of a spiral or wave")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L").resize((128, 128))
        st.image(img, caption="Uploaded Image", use_column_width=True)
        saved_path = save_temp_image(img)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌀 Detect Spiral"):
                run_detection(saved_path, SPIRAL_MODEL)

        with col2:
            if st.button("🌊 Detect Wave"):
                run_detection(saved_path, WAVE_MODEL)
