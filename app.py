from flask import Flask, render_template, Response
import cv2
import threading
from waste_sorting import process_frame

app = Flask(__name__)

# Initialize global variables
cap = None
thread_lock = threading.Lock()

# Function to generate video frames for streaming
def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process the current frame
        try:
            processed_frame, detected_items = process_frame(frame)  # Process frame
            # Encode the frame to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error during frame processing: {e}")
            break

    cap.release()

# Flask route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)