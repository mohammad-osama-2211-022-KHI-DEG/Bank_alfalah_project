import logging
import pickle
from datetime import datetime, timedelta, timezone

import cv2
import face_recognition
import requests
from mtcnn import MTCNN

# Constants
FILENAME = "area_manager"
MTCNN_CONFIDENCE = 0.97
TOLERANCE = 0.5
LOG_FILENAME = (
    f"/home/zubair/Xloop/bafl/face_recognition/area_manager/output/logs/{FILENAME}.log"
)
VIDEO_PATH = (
    f"/home/zubair/Xloop/bafl/face_recognition/branch_manager/video/{FILENAME}.mp4"
)
OUTPUT_VIDEO = f"/home/zubair/Xloop/bafl/face_recognition/area_manager/output/{FILENAME}_tol_{TOLERANCE}.avi"
ENCODINGS_FILE = (
    "/home/zubair/Xloop/bafl/face_recognition/area_manager/combined_images.pkl"
)
ALERT_API_URL = "http://13.126.160.174:5000/area-manager-visits"
NOTIFICATION_URL = "http://13.126.160.174:5000/notification"
API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU",  # Replace with your JWT token
    "X-SERVER-TO-SERVER": "true",
}

logging.basicConfig(
    filename=LOG_FILENAME,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Dictionary to store recognized individuals and their recognition status

# Setup logging
logger = logging.getLogger("Area Manager")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("logs/area_manager.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

recognized_individuals = set()
gender_dict = {
    "MALE": ["Mehmaam", "Faraz", "Zubair", "Rehman"],
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
            "visitTimestamp": datetime.now(timezone(timedelta(hours=5))).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            + str("+05:00"),
        }
        # update_gender(data, gender_dict)
        logging.info(f"Area Manager Data Request_payload: {data}")
        response = requests.post(
            f"{ALERT_API_URL}?areaManagerId=14&branchId=1",
            json=data,
            headers=API_HEADERS,
        )
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
            "usecase": "area manager",
        }
        logging.info(f"Notification Request_payload:{data}")
        # response = requests.post(NOTIFICATION_URL, json=data, headers=API_HEADERS)
        # if response is not None:
        #     if response and response.status_code // 100 == 2:
        #             # Handle successful response
        #             logging.info("Request Successful\nResponse: %s", response.text)
        #     else:
        #             logging.info(f"Request failed with status code {response.status_code}")
        # else:
        #     pass
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        logging.error("Failed to get a response")


def recognize_faces(frame, face_detector, encodings, names):
    logger.debug("Processing frame in Area Manager")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)

    for face in faces:
        if face["confidence"] > MTCNN_CONFIDENCE:
            x, y, w, h = face["box"]
            face_encodings = face_recognition.face_encodings(
                frame, [(y, x + w, y + h, x)]
            )
            if face_encodings:
                face_encodings = face_encodings[0]
                matches = face_recognition.compare_faces(
                    encodings, face_encodings, tolerance=TOLERANCE
                )
                if any(matches):
                    matched_idxs = [i for (i, b) in enumerate(matches) if b]
                    for i in matched_idxs:
                        name = names[i]
                        cv2.putText(
                            frame,
                            name,
                            (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            8,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            name,
                            (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 2)
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
    # frame_skip_count = 0
    # skip_interval = 15
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        # frame_skip_count +=1
        # if frame_skip_count % skip_interval != 0:
        #     continue
        recognize_faces(frame, face_detector, data["encodings"], data["names"])

        out.write(frame)
        cv2.imshow(f"{FILENAME}_tolerlance_{TOLERANCE}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
