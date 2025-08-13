# Parkinson’s Detector

**A Flask-based web app to predict Parkinson’s disease using ML models**

## Overview

This project provides a lightweight web application for detecting Parkinson’s disease using a trained machine learning model. Users can input relevant data via a web interface, and the app returns predictions using pre-trained models stored in the `models/` directory.

## Project Structure

```
├── .devcontainer/           # Configuration files for development containers (e.g., VS Code Remote Containers)  
├── models/                  # Pre-trained model files (pickle, HDF5, joblib, etc.)  
├── app.py                   # Main Flask application script  
├── predict_parkinsons.py    # Python module for data preprocessing and model predictions  
├── requirements.txt         # List of required Python dependencies  
├── runtime.txt              # Runtime specification (e.g., for deployment platforms like Heroku)  
└── README.md                # Project documentation  
```

## Installation & Setup

1. Clone the repository

```
git clone https://github.com/Shreya-Reddy2004/parkinsons-detector.git  
cd parkinsons-detector  
```

2. Create and activate a virtual environment (optional but recommended)

```
python3 -m venv venv  
source venv/bin/activate         # Linux / macOS  
venv\Scripts\activate            # Windows  
```

3. Install dependencies

```
pip install -r requirements.txt  
```

4. Ensure that your models are correctly placed in the `models/` directory (e.g., `model.pkl` or `model.joblib`).

## Running the Application

```
python app.py  
```

Once running, open a web browser and navigate to `http://127.0.0.1:5000` to access the prediction form.

## Project Components

* **app.py** – Initializes Flask, defines routes, renders HTML templates, and handles form submissions.
* **predict\_parkinsons.py** – Loads the trained model, processes input features, and generates predictions.
* **models/** – Contains the trained model file(s).

## Usage Example

1. Start the server:

```
python app.py  
```

2. Open `http://127.0.0.1:5000` in your browser.
3. Fill out the form with patient feature data.
4. Submit to receive a prediction: “Parkinson’s detected” or “No Parkinson’s”.

## Dependencies

* Flask
* scikit-learn
* pandas, numpy
* Optional: gunicorn or waitress for deployment

## Deployment Guidance

You can deploy this app to Heroku, Railway, or Render:

* Include a `Procfile` if required.
* Ensure `requirements.txt` is updated.
* Set the correct Python version in `runtime.txt`.

## Customization Ideas

* Improve the ML model with more data or algorithms.
* Add visualizations or SHAP explainability.
* Enhance UI using Bootstrap or similar frameworks.
* Create a REST API for predictions.
* Add automated testing.

## License

Specify your license here (MIT recommended for open use).

## Contact

Created by **Shreya Reddy** – Open for collaboration, questions, and feedback.
