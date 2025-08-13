
# Parkinson’s Detector

**A Streamlit-based web app to predict Parkinson’s disease from spiral or wave drawings using ML models**

## Overview

This project provides an interactive web application that helps detect potential signs of Parkinson’s disease by analyzing spiral or wave patterns. Users can either **draw directly on an in-browser canvas** or **upload an image** of a spiral/wave. The app uses pre-trained Random Forest models to classify results as **Healthy** or **Parkinson’s**.

## Project Structure

```
├── models/                  # Pre-trained model files (Random Forest for spiral & wave)  
├── app.py                   # Main Streamlit application script  
├── predict_parkinsons.py    # Functions to preprocess and predict using the models  
├── requirements.txt         # Python dependencies  
└── README.md                # Project documentation  
```

## Installation & Setup

1. **Clone the repository**

```
git clone https://github.com/Shreya-Reddy2004/parkinsons-detector.git  
cd parkinsons-detector  
```

2. **Create and activate a virtual environment (optional)**

```
python -m venv venv  
source venv/bin/activate      # Linux / macOS  
venv\Scripts\activate         # Windows  
```

3. **Install dependencies**

```
pip install -r requirements.txt  
```

## Running the Application

Run the following command:

```
streamlit run app.py  
```

Streamlit will open a browser window (usually at `http://localhost:8501`) with the app.

## How It Works

* **Draw on Canvas**:

  * Choose brush size and color from the sidebar.
  * Draw a spiral or wave on the provided canvas.
  * Click “🌀 Detect Spiral” or “🌊 Detect Wave” to run predictions.

* **Upload Image**:

  * Upload a PNG/JPG/JPEG image of a spiral or wave.
  * Click the respective detection button.

The app then:

1. Converts the image to grayscale.
2. Processes it for feature extraction.
3. Loads the respective trained model from `models/`.
4. Displays the prediction result: **🟢 Healthy** or **🔴 Parkinson’s**.

## Dependencies

* streamlit
* streamlit-drawable-canvas
* Pillow
* numpy
* opencv-python
* scikit-learn

(Exact versions are listed in `requirements.txt`)

## Deployment

You can deploy the app to **Streamlit Cloud** or other platforms:

* For Streamlit Cloud: Push to GitHub, sign in to Streamlit Cloud, and select the repo.
* Ensure `requirements.txt` contains all dependencies.

## Customization Ideas

* Improve the models with more training data.
* Add probability scores alongside predictions.
* Include SHAP or Grad-CAM visual explanations.
* Enhance UI with more theme options.

## Contact

Created by **Shreya Reddy** — open for collaboration and feedback.

