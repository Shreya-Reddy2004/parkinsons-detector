
# 🧠 Parkinson’s Detector

**Draw or upload spiral/wave images to detect potential Parkinson’s disease — powered by Machine Learning and Streamlit!**

---

## 📌 Overview

The **Parkinson’s Detector** is a web-based application built with **Streamlit** that allows users to either **draw** or **upload** images of **spirals** or **waves**.
It uses **Random Forest Classifier models** trained on medical datasets to predict whether the input indicates **Healthy** or **Parkinson’s** patterns.

This tool aims to provide a **quick, non-invasive pre-screening method** for research and educational purposes — **not as a substitute for professional medical diagnosis**.

---

## 📂 Project Structure

```
├── app.py                          # Main Streamlit application
├── predict_parkinsons.py            # Image preprocessing & prediction logic
├── models/
│   ├── random_forest_spiral_model.pkl
│   └── random_forest_wave_model.pkl
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## ✨ Features

* ✏ **Draw on Canvas**: Create spiral or wave drawings directly in the browser.
* 📤 **Upload Image**: Upload existing spiral/wave images for analysis.
* 🔍 **Two Prediction Modes**:

  * 🌀 Spiral detection
  * 🌊 Wave detection
* 🖼 **Live Preview** of your drawing or uploaded image.
* ⚡ **Fast Prediction** using pre-trained Random Forest models.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Shreya-Reddy2004/parkinsons-detector.git
cd parkinsons-detector
```

### 2️⃣ Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 🎯 How to Use

### **1. Draw on Canvas**

* Select **"Draw on Canvas"** from the sidebar.
* Adjust brush **stroke width** and **color**.
* Draw either a **spiral** or **wave**.
* Click **"🌀 Detect Spiral"** or **"🌊 Detect Wave"** to get the prediction.

### **2. Upload Image**

* Select **"Upload Image"** from the sidebar.
* Choose a PNG/JPG/JPEG file.
* Click **"🌀 Detect Spiral"** or **"🌊 Detect Wave"** to get the prediction.

---

## 🧠 Model Details

* **Algorithm**: Random Forest Classifier
* **Input**: Grayscale spiral or wave image
* **Preprocessing**:

  * Convert to grayscale
  * Resize and threshold image
* **Output**:

  * 🟢 Healthy
  * 🔴 Parkinson’s

---

## 📦 Dependencies

* `streamlit`
* `streamlit-drawable-canvas`
* `Pillow`
* `numpy`
* `opencv-python`
* `scikit-learn`

> 📜 Full list with versions is available in `requirements.txt`.

---

## ⚠️ Disclaimer

This project is **for research and educational purposes only**.
It is **not a diagnostic tool** and should not be used as a substitute for professional medical advice or examination.
