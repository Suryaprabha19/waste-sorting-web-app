import cv2
import numpy as np

# Initialize category labels and emission factors
categories = ["Recyclable", "Organic", "Non-Recyclable"]
emission_factors = {"Recyclable": 1.5, "Organic": 0.2, "Non-Recyclable": 3.0}

# Placeholder function for classification (replace with trained model later)
def classify_waste(image):
    """
    Simulate waste classification. Replace this with a trained ML model.
    Args:
        image (numpy array): Preprocessed input image.
    Returns:
        str: Predicted category of the waste item.
    """
    # Simulate classification with dummy logic (update with your model)
    avg_pixel_value = np.mean(image)
    if avg_pixel_value > 200:
        return "Recyclable"
    elif avg_pixel_value > 100:
        return "Organic"
    else:
        return "Non-Recyclable"

# Carbon footprint calculation
def calculate_carbon_footprint(detected_items):
    """
    Calculate total carbon footprint based on detected items.
    Args:
        detected_items (list): List of detected waste categories.
    Returns:
        float: Total carbon footprint (kg CO2-eq).
    """
    total_emissions = 0
    for item in detected_items:
        total_emissions += emission_factors.get(item, 0)
    return total_emissions

# Process frame for waste detection and classification
def process_frame(frame):
    """
    Process a single frame for waste detection and classification.
    Args:
        frame (numpy array): Input frame from video feed.
    Returns:
        numpy array: Annotated frame with waste categories.
        list: Detected waste categories in the frame.
    """
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

# Main program
def main():
    # Initialize video capture (webcam)
    cap = cv2.VideoCapture(0)

    # Store detected items for carbon footprint calculation
    all_detected_items = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Process frame
        processed_frame, detected_items = process_frame(frame)
        all_detected_items.extend(detected_items)

        # Display carbon footprint on the frame
        total_emissions = calculate_carbon_footprint(all_detected_items)
        cv2.putText(processed_frame, f"Carbon Footprint: {total_emissions:.2f} kg CO2-eq",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show the processed frame
        cv2.imshow("Waste Sorting", processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    print("\nDetected Items Summary:")
    print(all_detected_items)
    total_emissions = calculate_carbon_footprint(all_detected_items)
    print(f"Total Carbon Footprint: {total_emissions:.2f} kg CO2-eq")

if __name__ == "__main__":
    main()
