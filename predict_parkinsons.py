import os
import cv2
import joblib
import numpy as np
from skimage import feature

def quantify_image(image):
    """
    Extract Histogram of Oriented Gradients (HOG) features from the image.
    """
    features = feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L1"
    )
    return features

def predict_image(img, mdl):
    """
    Predict whether the image indicates Parkinson's using the specified model.

    Parameters:
    - img (str): Path to the input image.
    - mdl (str): Path to the trained model file.

    Returns:
    - str: "Parkinsons" or "Healthy"
    """
    if not os.path.exists(img):
        raise FileNotFoundError(f"Image file '{img}' does not exist.")

    image = cv2.imread(img)
    if image is None:
        raise ValueError(f"Failed to load image from '{img}'. Ensure it's a valid image.")

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Extract features and make prediction
    features = quantify_image(image)
    model = joblib.load(mdl)
    preds = model.predict([features])

    return "Parkinsons" if preds[0] else "Healthy"
