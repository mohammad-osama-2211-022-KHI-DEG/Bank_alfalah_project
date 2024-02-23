from datetime import datetime, timedelta, timezone

import cv2
import face_recognition
import numpy as np
import requests
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN

# Constants
DISTANCE_THRESHOLD = 0.7
CLASS_LABELS = ["Angry", "Not Angry", "Not Angry", "Not Angry", "Not Angry"]
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
VIDEO_FILE_PATH = "anger.mp4"
EMOTION_DETECTION_MODEL_PATH = "./Emotion_Detection.h5"
SERVER_URL = "http://13.235.71.140:5000"

# Initialize a dictionary for storing face encodings and IDs
face_encodings_dict = {}
# Initialize a dictionary to store previous emotions for each unique face
previous_emotions = {}
# Initialize a set to store unique face IDs that are happy
happy_face_ids = set()
# Initialize the total happy count
Angry_count = 0
# Initialize a counter for face IDs
face_id_counter = 0

# Initialize the previous unique happy count
previous_unique_Angry_count = 0
# Initialize the previous footfall number
previous_footfall_number = 0

# Initialize a counter for notifications
notification_counter = 0

# Set the desired window size
window_width = 1000
window_height = 1000

# Open a video capture object
video_capture = cv2.VideoCapture(VIDEO_FILE_PATH)

# Load emotion detection model
classifier = load_model(EMOTION_DETECTION_MODEL_PATH)


# Function to send notification
def send_notification(message, usecase):
    url = f"{SERVER_URL}/notification"
    data = {
        "timestamp": datetime.now(timezone(timedelta(hours=5))).strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        ),
        "message": message,
        "country": "pakistan",
        "city": "karachi",
        "branch": "clifton",
        "usecase": usecase,
    }
    response = requests.post(url, json=data, headers=get_request_headers())
    return response


# Function to send emotion data
def send_emotion_data(Angry_count, unique_Angry_count, footfall_number):
    url = f"{SERVER_URL}/emotion/aggressive"
    headers = get_request_headers()
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    data = {
        # "happyCount": Angry_count,
        "noOfAggressivePeople": unique_Angry_count,
        "footfallNumber": footfall_number,
        "timestamp": current_datetime.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "country": "pakistan",
        "city": "karachi",
        "branch": "clifton",
    }
    response = requests.post(url, json=data, headers=headers)
    return response


# Function to get request headers
def get_request_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJuYXF2aXNheWFtM0BnbWFpbC5jb20iLCJhdXRob3JpdGllcyI6IkNSRUFURV9VU0VSLFZJRVciLCJpYXQiOjE3MDYwNDE0NjMsImV4cCI6MTcwNjM0MTQ2M30.EGPkhpAzqZ-vL-1J4RBYoGn4N1VO9bWQqNa3W4wXEjI",  # Replace with your JWT token
        "X-SERVER-TO-SERVER": "true",
    }


# Initialize face detection
detector = MTCNN()

while True:
    Angry_face_ids_current_frame = set()
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face["box"]
        x, y = abs(x), abs(y)

        # Extract face location and encoding
        face_location = [(y, x + w, y + h, x)]
        face_encodings = face_recognition.face_encodings(frame, face_location)

        # Check if the face is recognized
        matched_id = None
        min_distance = float("inf")

        for person_id, encoding in face_encodings_dict.items():
            distance = face_recognition.face_distance([encoding], face_encodings[0])[0]
            if distance < DISTANCE_THRESHOLD and distance < min_distance:
                matched_id = person_id
                min_distance = distance

        if matched_id is not None:
            # Face recognized
            recognized_person_id = matched_id
        else:
            # New face detected
            recognized_person_id = f"Face {face_id_counter}"
            face_id_counter += 1
            face_encodings_dict[recognized_person_id] = face_encodings[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame[y : y + h, x : x + w]
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classifier.predict(roi)[0]
            label = CLASS_LABELS[preds.argmax()]
            label_position = (
                x,
                y - 10,
            )  # Adjust the position to be just above the face
            cv2.putText(
                frame,
                f"{recognized_person_id} - {label}",
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Handle emotions
            if label == "Angry":
                if recognized_person_id not in Angry_face_ids_current_frame:
                    Angry_face_ids_current_frame.add(recognized_person_id)

                if previous_emotions.get(recognized_person_id, "") != "Angry":
                    # Only increment happy_count when the label changes from not happy to happy
                    Angry_face_ids_current_frame.add(recognized_person_id)
                    Angry_count += 1

            # Update the previous emotion for the person
            previous_emotions[recognized_person_id] = label

    # Check if there are changes in happy faces or footfall
    if (len(Angry_face_ids_current_frame) != previous_unique_Angry_count) or (
        len(faces) != previous_footfall_number
    ):
        response = send_emotion_data(
            Angry_count, len(Angry_face_ids_current_frame), len(faces)
        )
        print(response.status_code, response.content)

    previous_unique_happy_count = len(Angry_face_ids_current_frame)
    previous_footfall_number = len(faces)

    # Send notifications
    if len(Angry_face_ids_current_frame) >= 1:
        response = send_notification(
            f"number of unique Angry count is {len(Angry_face_ids_current_frame)}",
            "Agressive_customers",
        )
        print(response.status_code, response.content)

        if response.status_code == 200:
            notification_counter += 1

    frame = cv2.resize(frame, (window_width, window_height))

    cv2.putText(
        frame,
        f"Angry Count: {Angry_count}",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Unique Angry Count: {len(Angry_face_ids_current_frame)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"foot_fall number: {len(faces)}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Emotion Detector Islamabad Branch", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
