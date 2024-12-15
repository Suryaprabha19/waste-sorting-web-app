import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and categories
MODEL_PATH = "my_wastemodel.h5"  # Path to your model file
model = load_model(MODEL_PATH)
categories = ["Recyclable", "Organic", "Non-Recyclable"]  # Ensure this matches your model classes

# Preprocess image for model input
def preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Classify waste using the loaded model
def classify_waste(image):
    preprocessed = preprocess_image(image)
    predictions = model.predict(preprocessed)
    predicted_label = categories[np.argmax(predictions)]
    return predicted_label

# Process frame for waste detection and classification
def process_frame(frame):
    detected_items = []

    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]

            # Classify waste in the region of interest
            category = classify_waste(roi)
            detected_items.append(category)

            # Draw bounding box and label
            color = (0, 255, 0) if category == "Recyclable" else (255, 0, 0) if category == "Organic" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, detected_items