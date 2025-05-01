import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
import joblib
from skimage import feature
import cv2
from predict_parkinsons import predict_image

# Paths to the models
SPIRAL_MODEL = os.path.join("models", "random_forest_spiral_model.pkl")
WAVE_MODEL = os.path.join("models", "random_forest_wave_model.pkl")

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def explain_prediction(image_path, model_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    model = joblib.load(model_path)
    explainer = shap.Explainer(model.predict, masker=features.reshape(1, -1))
    shap_values = explainer(features.reshape(1, -1))
    return shap_values.values[0], features, explainer

st.title("\U0001F9E0 Parkinson's Detector with SHAP Explanation")

st.sidebar.title("Choose Input Method")
input_mode = st.sidebar.radio("Select input type", ["Draw on Canvas", "Upload Image"])

st.sidebar.title("Brush Settings")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 10, 3)
stroke_color = st.sidebar.color_picker("Stroke color", "#000000")

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
        alpha = img_rgba[:, :, 3] > 0
        rgb_array = (img_rgba[:, :, :3] * 255).astype(np.uint8)
        white_bg = np.ones_like(rgb_array, dtype=np.uint8) * 255
        white_bg[alpha] = rgb_array[alpha]

        img = Image.fromarray(white_bg).convert("L")
        img.save("temp_canvas.png")

        st.image(img, caption="Your Drawing", use_column_width=False)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("\U0001F300 Detect Spiral"):
                pred = predict_image("temp_canvas.png", SPIRAL_MODEL)
                st.markdown(f"### Result: {'\U0001F7E2 Healthy' if pred == 'Healthy' else '\U0001F534 Parkinson’s'}")
                st.markdown("#### \U0001F50D SHAP Explanation")
                shap_values, features, explainer = explain_prediction("temp_canvas.png", SPIRAL_MODEL)
                st.markdown("This plot shows which gradient patterns (HOG features) influenced the model's prediction.")
                shap.plots.bar(shap_values, show=False)
                st.pyplot(plt.gcf())

        with col2:
            if st.button("\U0001F30A Detect Wave"):
                pred = predict_image("temp_canvas.png", WAVE_MODEL)
                st.markdown(f"### Result: {'\U0001F7E2 Healthy' if pred == 'Healthy' else '\U0001F534 Parkinson’s'}")
                st.markdown("#### \U0001F50D SHAP Explanation")
                shap_values, features, explainer = explain_prediction("temp_canvas.png", WAVE_MODEL)
                st.markdown("This plot shows which gradient patterns (HOG features) influenced the model's prediction.")
                shap.plots.bar(shap_values, show=False)
                st.pyplot(plt.gcf())

elif input_mode == "Upload Image":
    st.subheader("Upload an image of a spiral or wave")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img.save("temp_uploaded.png")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("\U0001F300 Detect Spiral"):
                pred = predict_image("temp_uploaded.png", SPIRAL_MODEL)
                st.markdown(f"### Result: {'\U0001F7E2 Healthy' if pred == 'Healthy' else '\U0001F534 Parkinson’s'}")
                st.markdown("#### \U0001F50D SHAP Explanation")
                shap_values, features, explainer = explain_prediction("temp_uploaded.png", SPIRAL_MODEL)
                st.markdown("This plot shows which gradient patterns (HOG features) influenced the model's prediction.")
                shap.plots.bar(shap_values, show=False)
                st.pyplot(plt.gcf())

        with col2:
            if st.button("\U0001F30A Detect Wave"):
                pred = predict_image("temp_uploaded.png", WAVE_MODEL)
                st.markdown(f"### Result: {'\U0001F7E2 Healthy' if pred == 'Healthy' else '\U0001F534 Parkinson’s'}")
                st.markdown("#### \U0001F50D SHAP Explanation")
                shap_values, features, explainer = explain_prediction("temp_uploaded.png", WAVE_MODEL)
                st.markdown("This plot shows which gradient patterns (HOG features) influenced the model's prediction.")
                shap.plots.bar(shap_values, show=False)
                st.pyplot(pl
