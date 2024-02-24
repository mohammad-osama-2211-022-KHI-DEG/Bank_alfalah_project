import logging
from datetime import datetime, timedelta, timezone

import cv2
import face_recognition
import numpy as np
import requests
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN

# Constants
FILENAME = "anger"
DISTANCE_THRESHOLD = 0.70
MTCNN_CONFIDENCE = 0.98
PROCESS_FRAMES_PER_SECOND = 25
NUM_JITTERS = 10
CLASS_LABELS = ["Angry", "Not Angry", "Not Angry", "Not Angry", "Not Angry"]
LOG_FILE_NAME = f"/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/aggressive/output/logs/{FILENAME}.log"
VIDEO_FILE_PATH = f"/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/aggressive/video/{FILENAME}.mp4"
OUTPUT_VIDEO_PATH = f"/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/aggressive/output/{FILENAME}_dist_{DISTANCE_THRESHOLD}.mkv"
EMOTION_DETECTION_MODEL_PATH = "/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/aggressive/Emotion_Detection.h5"
SERVER_URL = "http://13.126.160.174:5000"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true",
}

# Set up logging
logging.basicConfig(
    filename=LOG_FILE_NAME,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize a dictionary for storing face encodings and IDs
face_encodings_dict = {}
# Initialize a dictionary to store previous emotions for each unique face
previous_emotions = {}
# Initialize the total angry count
angry_count = 0
# Initialize a counter for face IDs
face_id_counter = 0
# Initialize the previous unique angry count
previous_unique_angry_count = 0
# Initialize the previous footfall number
previous_footfall_number = 0
# Initialize a counter for notifications
notification_counter = 0

# Open a video capture object
video_capture = cv2.VideoCapture(VIDEO_FILE_PATH)
# Set the desired window size
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
logging.info(f"fps: {fps}")
# Load emotion detection model
classifier = load_model(EMOTION_DETECTION_MODEL_PATH)


# Function to send notification
def send_notification(message, usecase):
    url = f"{SERVER_URL}/notification"
    try:
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
        logging.info(f"Notification Request_payload:{data}")
        response = requests.post(url, json=data, headers=HEADERS)
        return response
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        return None


# Function to send emotion data
def send_emotion_data(unique_angry_count, footfall_number):
    url = f"{SERVER_URL}/emotion/aggressive"
    try:
        data = {
            "noOfAggressivePeople": unique_angry_count,
            "footfallNumber": footfall_number,
            "timestamp": datetime.now(timezone(timedelta(hours=5))).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            + str("+05:00"),
            "country": "pakistan",
            "city": "karachi",
            "branch": "clifton",
        }
        logging.info(f"Aggressive Data Request_payload:{data}")
        response = requests.post(url, json=data, headers=HEADERS)
        return response
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        return None


# Initialize face detection
detector = MTCNN()
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, FOURCC, fps, (frame_width, frame_height))

process_interval = int(round(fps / PROCESS_FRAMES_PER_SECOND))
logging.info(f"process_interval: {process_interval}")
while True:
    # Initialize a set to store unique face IDs that are angry
    angry_face_ids_current_frame = set()
    foot_fall_ids_current_frame = set()
    ret, frame = video_capture.read()

    if not ret:
        break

    # Determine the current frame number
    current_frame_number = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))

    # Process only the selected frames based on the interval
    if current_frame_number % process_interval == 0:
        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)
        for face in faces:
            confidence = face["confidence"]
            x, y, w, h = face["box"]
            x, y = abs(x), abs(y)
            if confidence > MTCNN_CONFIDENCE:
                # Extract face location and encoding
                face_location = [(y, x + w, y + h, x)]
                face_encodings = face_recognition.face_encodings(
                    frame, face_location, num_jitters=NUM_JITTERS
                )
                # Check if the face is recognized
                matched_id = None

                for person_id, encoding in face_encodings_dict.items():
                    distance = face_recognition.face_distance(
                        [encoding], face_encodings[0]
                    )[0]
                    if distance < DISTANCE_THRESHOLD:
                        matched_id = person_id

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
                        1,
                        (0, 0, 0),
                        10,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"{recognized_person_id} - {label}",
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    # current foot fall
                    if recognized_person_id is not None:
                        foot_fall_ids_current_frame.add(recognized_person_id)

                    # Handle emotions
                    if label == "Angry":
                        if recognized_person_id not in angry_face_ids_current_frame:
                            angry_face_ids_current_frame.add(recognized_person_id)
                        if previous_emotions.get(recognized_person_id, "") != "Angry":
                            # Only increment angry_count when the label changes from not angry to angry
                            angry_face_ids_current_frame.add(recognized_person_id)
                            angry_count += 1
                    # Update the previous emotion for the person
                    previous_emotions[recognized_person_id] = label

            # Check if there are changes in angry faces or footfall
            if (len(angry_face_ids_current_frame) != previous_unique_angry_count) or (
                len(foot_fall_ids_current_frame) != previous_footfall_number
            ):
                response = send_emotion_data(
                    len(angry_face_ids_current_frame), len(foot_fall_ids_current_frame)
                )
                if response is not None:
                    if response and response.status_code == 200:
                        logging.info("Request Successful\nResponse:", response.text)
                    else:
                        logging.info(
                            f"Request failed with status code {response.status_code}"
                        )
                    previous_unique_angry_count = len(angry_face_ids_current_frame)
                    previous_footfall_number = len(foot_fall_ids_current_frame)
                else:
                    logging.error("Failed to get a response")

            # Send notifications
            if (
                len(angry_face_ids_current_frame) >= 1
                and len(angry_face_ids_current_frame) % 1 == 0
                and len(angry_face_ids_current_frame) // 1 > notification_counter
            ):
                response = send_notification(
                    f"number of unique angry count is {len(angry_face_ids_current_frame)}",
                    "angry_customers",
                )
                if response is not None:
                    if response and response.status_code == 200:
                        logging.info("Request Successful\nResponse:", response.text)
                        notification_counter += 1
                    else:
                        logging.info(
                            f"Request failed with status code {response.status_code}"
                        )
                else:
                    logging.error("Failed to get a response")
    # Write information about Angry Counts, Unique Angry Counts and Foot_fall Number
    cv2.putText(
        frame,
        f"Angry Count: {angry_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        10,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Angry Count: {angry_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Unique Angry Count: {len(angry_face_ids_current_frame)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        10,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Unique Angry Count: {len(angry_face_ids_current_frame)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"foot_fall number: {len(foot_fall_ids_current_frame)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        10,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"foot_fall number: {len(foot_fall_ids_current_frame)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    out.write(frame)
    cv2.imshow(f"{FILENAME}_dist_{DISTANCE_THRESHOLD}", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
out.release()
cv2.destroyAllWindows()
