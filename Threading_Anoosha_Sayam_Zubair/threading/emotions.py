import logging
from datetime import datetime, timedelta, timezone

import cv2
import face_recognition
import numpy as np
import requests
from input_stream import *
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN

# Constants
MTCNN_CONFIDENCE = 0.97
DISTANCE_THRESHOLD = 0.6
CLASS_LABELS = ["Frustrated", "Happy", "not Happy", "not Happy", "not Happy"]
VIDEO_FILE_PATH = "smile.mp4"
EMOTION_DETECTION_MODEL_PATH = "Emotion_Detection.h5"
SERVER_URL = "http://13.126.160.174:5000"

# # Set up logging
logging.basicConfig(
    filename="logs/thread_precessing_combined.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Setup logging
logger = logging.getLogger("Emotions")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("logs/emotions.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


# Initialize global variables
face_encodings_dict = {}
previous_emotions = {}
happy_face_ids = set()
anngery_face_ids = set()
happy_count = 0
Frustrated_count = 0
previous_unique_Frustrated_count = 0

face_id_counter = 0
notification_counter = 0
# Load emotion detection model
classifier = load_model(EMOTION_DETECTION_MODEL_PATH)


# Function to send notification
def send_notification(message, usecase):
    url = f"{SERVER_URL}/notification"
    data = {
        "timestamp": datetime.now(timezone(timedelta(hours=5))).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + str("+05:00"),
        "message": message,
        "country": "pakistan",
        "city": "karachi",
        "branch": "clifton",
        "usecase": usecase,
    }
    response = requests.post(url, json=data, headers=get_request_headers())
    return response


# Function to send emotion data
def send_emotion_data(happy_count, unique_smile_count, footfall_number):
    url = f"{SERVER_URL}/emotion/happy"
    headers = get_request_headers()
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    data = {
        "happyCount": happy_count,
        "noOfUniqueSmile": unique_smile_count,
        "footfallNumber": footfall_number,
        "timestamp": current_datetime.strftime("%Y-%m-%dT%H:%M:%S") + str("+05:00"),
        "country": "pakistan",
        "city": "karachi",
        "branch": "clifton",
    }
    response = requests.post(url, json=data, headers=headers)
    return response


def send_emotion_frustrated_data(
    Frustrated_count, unique_Frustrated_count, footfall_number
):
    url = f"{SERVER_URL}/emotion/aggressive"
    headers = get_request_headers()
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    data = {
        "noOfAggressivePeople": unique_Frustrated_count,
        "footfallNumber": footfall_number,
        "timestamp": current_datetime.strftime("%Y-%m-%dT%H:%M:%S") + str("+05:00"),
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
        "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU",  # Replace with your JWT token
        "X-SERVER-TO-SERVER": "true",
    }


# Initialize face detection
detector = MTCNN()


# Main function to process video frames
def main1():
    for frame in capture_webcam():
        processed_data = process_frame(frame)
        if processed_data is None:
            break
        show_info(frame, *processed_data)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# Function to detect faces using MTCNN
def detect_faces(frame):
    return detector.detect_faces(frame)


# Function to handle each detected face
def handle_face(
    face,
    frame,
    happy_face_ids_current_frame,
    Frustrated_face_ids_current_frame,
    foot_fall_ids_current_frame,
):
    global face_id_counter, happy_count, Frustrated_count
    if face["confidence"] > MTCNN_CONFIDENCE:
        x, y, w, h = face["box"]
        x, y = abs(x), abs(y)
        face_location = [(y, x + w, y + h, x)]
        face_encodings = face_recognition.face_encodings(frame, face_location)
        matched_id = None
        min_distance = float("inf")
        for person_id, encoding in face_encodings_dict.items():
            distance = face_recognition.face_distance([encoding], face_encodings[0])[0]
            if distance < DISTANCE_THRESHOLD and distance < min_distance:
                matched_id = person_id
                min_distance = distance
        if matched_id is not None:
            recognized_person_id = matched_id
        else:
            recognized_person_id = f"Face {face_id_counter}"
            face_id_counter += 1
            face_encodings_dict[recognized_person_id] = face_encodings[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = classify_emotion(frame, (x, y, w, h))

        # current foot fall
        if recognized_person_id is not None:
            foot_fall_ids_current_frame.add(recognized_person_id)

        if label == "Happy":
            if recognized_person_id not in happy_face_ids_current_frame:
                happy_face_ids_current_frame.add(recognized_person_id)
            if previous_emotions.get(recognized_person_id, "") != "Happy":
                happy_face_ids_current_frame.add(recognized_person_id)
                happy_count += 1

        if label == "Frustrated":
            if recognized_person_id not in Frustrated_face_ids_current_frame:
                Frustrated_face_ids_current_frame.add(recognized_person_id)

            if previous_emotions.get(recognized_person_id, "") != "Frustrated":
                # Only increment happy_count when the label changes from not happy to happy
                Frustrated_face_ids_current_frame.add(recognized_person_id)
                Frustrated_count += 1

        label_position = (x, y - 10)
        previous_emotions[recognized_person_id] = label
        cv2.putText(
            frame,
            f"{recognized_person_id} - {label}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            8,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{recognized_person_id} - {label}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )


# Function to classify emotion using the loaded model
def classify_emotion(frame, face_coords):
    x, y, w, h = face_coords
    roi_gray = frame[y : y + h, x : x + w]
    roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = classifier.predict(roi)[0]
        label = CLASS_LABELS[preds.argmax()]
        return label


# Function to handle emotion detection and sending data
def handle_emotions(
    happy_face_ids_current_frame, footfall_number, Frustrated_face_ids_current_frame
):
    global happy_count, notification_counter
    previous_footfall_number = 0
    if (len(happy_face_ids_current_frame) != len(happy_face_ids)) or (
        footfall_number != len(face_encodings_dict)
    ):
        response = send_emotion_data(
            happy_count, len(happy_face_ids_current_frame), footfall_number
        )
        # print(response.status_code, response.content)
        if response is not None:
            if response and response.status_code == 200:
                logging.info(f"Request Successful for Data\nResponse:{response.text}")
            else:
                logging.info(f"Request failed with status code {response.status_code}")
        else:
            logging.error("Failed to get a response")
    happy_face_ids.clear()
    happy_face_ids.update(happy_face_ids_current_frame)
    if (
        len(happy_face_ids_current_frame) >= 5
        and len(happy_face_ids_current_frame) % 5 == 0
        # and len(happy_face_ids_current_frame) // 5 > notification_counter
    ):
        response = send_notification(
            f"number of unique happy count is {len(happy_face_ids_current_frame)}",
            "happy_customers",
        )
        if response is not None:
            if response and response.status_code == 200:
                logging.info(
                    f"Request Successful for Notification\nResponse:{response.text}"
                )
                # notification_counter += 1
            else:
                logging.info(f"Request failed with status code {response.status_code}")
        else:
            logging.error("Failed to get a response")

    if (len(Frustrated_face_ids_current_frame) != previous_unique_Frustrated_count) or (
        footfall_number != previous_footfall_number
    ):
        response = send_emotion_frustrated_data(
            Frustrated_count, len(Frustrated_face_ids_current_frame), footfall_number
        )

    previous_unique_happy_count = len(Frustrated_face_ids_current_frame)
    previous_footfall_number = footfall_number


def process_frame(frame):
    logger.debug("Processing frame in Emotions")
    happy_face_ids_current_frame = set()
    Frustrated_face_ids_current_frame = set()
    foot_fall_ids_current_frame = set()
    faces = detect_faces(frame)

    for face in faces:
        handle_face(
            face,
            frame,
            happy_face_ids_current_frame,
            Frustrated_face_ids_current_frame,
            foot_fall_ids_current_frame,
        )
        handle_emotions(
            happy_face_ids_current_frame,
            len(foot_fall_ids_current_frame),
            Frustrated_face_ids_current_frame,
        )
        show_info(
            frame,
            happy_count,
            len(happy_face_ids_current_frame),
            len(Frustrated_face_ids_current_frame),
            len(foot_fall_ids_current_frame),
        )
    return (
        happy_count,
        len(happy_face_ids_current_frame),
        len(Frustrated_face_ids_current_frame),
        len(foot_fall_ids_current_frame),
    )


# Function to display information on frames
def show_info(
    frame, happy_count, unique_smile_count, unique_Frustrated_count, footfall_number
):
    cv2.putText(
        frame,
        f"Happy Count: {happy_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Happy Count: {happy_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Unique Happy Count: {unique_smile_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Unique Happy Count: {unique_smile_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Unique Frustrated Counts: {unique_Frustrated_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Unique Frustrated Counts: {unique_Frustrated_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Footfall Number: {footfall_number}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Footfall Number: {footfall_number}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    # cv2.imshow('Emotion Detector', frame)


if __name__ == "__main__":
    main1()
