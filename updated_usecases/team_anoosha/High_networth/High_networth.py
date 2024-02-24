import logging
import pickle
from datetime import datetime, timedelta, timezone

import cv2
import face_recognition
import requests
from mtcnn import MTCNN

# Constants
FILENAME = "rehman_HNW"
TOLERANCE = 0.30
LOG_FILENAME = f"/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/high_value_customer/output/logs/{FILENAME}.log"
VIDEO_PATH = f"/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/high_value_customer/video/{FILENAME}.mp4"
OUTPUT_VIDEO = f"/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/high_value_customer/output/{FILENAME}_tol_{TOLERANCE}.avi"
ENCODINGS_FILE = "/home/zubair/Xloop/bafl/ready_github_code/team_anoosha/high_value_customer/encodings/combined_images.pkl"
ALERT_API_URL = "http://13.126.160.174:5000/high-net-worth"
NOTIFICATION_URL = "http://13.126.160.174:5000/notification"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU"  # Replace with your JWT token
API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true",
}

logging.basicConfig(
    filename=LOG_FILENAME,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Dictionary to store recognized individuals and their recognition status
recognized_individuals = set()

# Assign gender for High Value Customers
gender_dict = {
    "MALE": ["Mehmaam", "Faraz", "Zubair", "Rehman", "Maqsood", "Ali"],
    "FEMALE": ["Novette", "Valery", "Anoosha"],
}


def load_encodings(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error(f"Encodings file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading encodings: {e}")
        return None


def update_gender(data, gender_dict):
    # Extracting the customerName from data
    name = data["customerName"]

    # Checking and updating gender based on the name
    for gender, names in gender_dict.items():
        if name in names:
            data["gender"] = gender
            break


def generate_alert(name):
    try:
        data = {
            "country": "pakistan",
            "branch": "clifton",
            "city": "karachi",
            "timestamp": datetime.now(timezone(timedelta(hours=5))).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            + str("+05:00"),
            "customerName": name,
        }
        update_gender(data, gender_dict)
        logging.info(f"High Networth Data Request_payload: {data}")
        response = requests.post(ALERT_API_URL, json=data, headers=API_HEADERS)
        if response is not None:
            # Check if the request was successful (status code 2xx)
            if response and response.status_code // 100 == 2:
                # Handle successful response
                logging.info("Alert sent successfully\nResponse: %s", response.text)
            else:
                logging.info(f"Request failed with status code {response.status_code}")
        else:
            pass
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        logging.error("Failed to get a response")

    try:
        data = {
            "timestamp": datetime.now(timezone(timedelta(hours=5))).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            + str("+05:00"),
            "message": f"{name} is detected in Clifton Branch, Karachi at {datetime.now(timezone(timedelta(hours=5))).strftime('%I:%M %p, %d-%b-%Y')}",
            "country": "pakistan",
            "city": "karachi",
            "branch": "clifton",
            "usecase": "high networth",
        }
        logging.info(f"Notification Request_payload:{data}")
        response = requests.post(NOTIFICATION_URL, json=data, headers=API_HEADERS)
        if response is not None:
            if response and response.status_code // 100 == 2:
                # Handle successful response
                logging.info("Request Successful\nResponse: %s", response.text)
            else:
                logging.info(f"Request failed with status code {response.status_code}")
        else:
            pass
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        logging.error("Failed to get a response")


def recognize_faces(frame, face_detector, data):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face["box"]
        encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])
        if encodings:
            encodings = encodings[0]
            matches = face_recognition.compare_faces(
                data["encodings"], encodings, tolerance=TOLERANCE
            )
            if any(matches):
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                for i in matched_idxs:
                    name = data["names"][i]
                    cv2.putText(
                        frame,
                        f"{name}",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )
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

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        recognize_faces(frame, face_detector, data)

        out.write(frame)
        cv2.imshow(f"{FILENAME}_tolerance_{TOLERANCE}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
