import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
from predict_parkinsons import predict_image
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Helper to display SHAP force plot in Streamlit
def st_shap(plot, height=400):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Model paths
SPIRAL_MODEL = os.path.join("models", "random_forest_spiral_model.pkl")
WAVE_MODEL = os.path.join("models", "random_forest_wave_model.pkl")

st.title("🧠 Parkinson's Detector")

with st.expander("ℹ️ How this works"):
    st.markdown("""
    This app detects Parkinson's symptoms from spiral or wave drawings using machine learning (Random Forest).
    
    - It extracts features from your drawing using **Histogram of Oriented Gradients (HOG)**.
    - It then makes a prediction and uses **SHAP** (SHapley Additive exPlanations) to explain how much each feature influenced the decision.
    - Features that increase the chance of Parkinson's are shown in **red**, and those supporting a healthy diagnosis in **blue**.
    """)

# Sidebar
st.sidebar.title("Choose Input Method")
input_mode = st.sidebar.radio("Select input type", ["Draw on Canvas", "Upload Image"])

st.sidebar.title("Brush Settings")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 10, 3)
stroke_color = st.sidebar.color_picker("Stroke color", "#000000")

# Canvas Drawing
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
        if col1.button("🌀 Detect Spiral"):
            label, features = predict_image("temp_canvas.png", SPIRAL_MODEL)
            st.markdown(f"### Result: {'🟢 Healthy' if label == 'Healthy' else '🔴 Parkinson’s'}")
            model = joblib.load(SPIRAL_MODEL)
            X = pd.DataFrame([features])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            st.subheader("Feature Contribution (Spiral)")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig)
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0]))

        if col2.button("🌊 Detect Wave"):
            label, features = predict_image("temp_canvas.png", WAVE_MODEL)
            st.markdown(f"### Result: {'🟢 Healthy' if label == 'Healthy' else '🔴 Parkinson’s'}")
            model = joblib.load(WAVE_MODEL)
            X = pd.DataFrame([features])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            st.subheader("Feature Contribution (Wave)")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig)
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0]))

# Upload Image
elif input_mode == "Upload Image":
    st.subheader("Upload an image of a spiral or wave")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img.save("temp_uploaded.png")

        col1, col2 = st.columns(2)
        if col1.button("🌀 Detect Spiral"):
            label, features = predict_image("temp_uploaded.png", SPIRAL_MODEL)
            st.markdown(f"### Result: {'🟢 Healthy' if label == 'Healthy' else '🔴 Parkinson’s'}")
            model = joblib.load(SPIRAL_MODEL)
            X = pd.DataFrame([features])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            st.subheader("Feature Contribution (Spiral)")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig)
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0]))

        if col2.button("🌊 Detect Wave"):
            label, features = predict_image("temp_uploaded.png", WAVE_MODEL)
            st.markdown(f"### Result: {'🟢 Healthy' if label == 'Healthy' else '🔴 Parkinson’s'}")
            model = joblib.load(WAVE_MODEL)
            X = pd.DataFrame([features])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            st.subheader("Feature Contribution (Wave)")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig)
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0]))
