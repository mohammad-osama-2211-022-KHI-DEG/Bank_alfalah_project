import cv2
from mtcnn import MTCNN
import face_recognition
import pickle
import logging
from datetime import datetime, timezone, timedelta 
import requests

# Constants
VIDEO_PATH = 'your_video'
ENCODINGS_FILE = 'face_enc'
OUTPUT_VIDEO = 'output/output.avi'
TOLERANCE = 0.3
current_datetime = datetime.now(timezone(timedelta(hours=5)))  # Adjust the timezone offset as needed

# API endpoint for sending alerts
ALERT_API_URL = 'http://13.235.71.140:5000/high-net-worth'
# Headers for API request
API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJuYXF2aXNheWFtM0BnbWFpbC5jb20iLCJhdXRob3JpdGllcyI6IkNSRUFURV9VU0VSLFZJRVciLCJpYXQiOjE3MDYwNDE0NjMsImV4cCI6MTcwNjM0MTQ2M30.EGPkhpAzqZ-vL-1J4RBYoGn4N1VO9bWQqNa3W4wXEjI",   # Replace with your JWT token
    "X-SERVER-TO-SERVER": "true"
}

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to store recognized individuals and their recognition status
recognized_individuals = set()

def load_encodings(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error(f"Encodings file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading encodings: {e}")
        return None

def generate_alert(name):
    data = {
        "country": "pakistan",
        "branch": "clifton",
        "city": "karachi",
        "timestamp": current_datetime.strftime('%Y-%m-%dT%H:%M:%S%z'),
        "customerName": name,
        "gender": "MALE"  
    }
    response = requests.post(ALERT_API_URL, json=data, headers=API_HEADERS)
    
    # Check if the request was successful (status code 2xx)
    if response.status_code // 100 == 2:
        logging.info("Alert sent successfully")
        print("Response:", response.text)
    else:
        logging.error(f"Failed to send alert. Status code: {response.status_code}")

def recognize_faces(frame, face_detector, data):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face['box']
        encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])
        if encodings:
            encodings = encodings[0]
            matches = face_recognition.compare_faces(data["encodings"], encodings, tolerance=TOLERANCE)
            if any(matches):
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                for i in matched_idxs:
                    name = data["names"][i]
                    cv2.putText(frame, f"{name}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Generate alert only the first time the person is detected
                    if name not in recognized_individuals:
                        generate_alert(name)
                        recognized_individuals.add(name)

def main():
    # Initialize MTCNN face detector
    face_detector = MTCNN()
    
    # Load face encodings
    data = load_encodings(ENCODINGS_FILE)
    if not data:
        return

    logging.info("Streaming started")
    video_capture = cv2.VideoCapture(VIDEO_PATH)

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        recognize_faces(frame, face_detector, data)

        out.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()