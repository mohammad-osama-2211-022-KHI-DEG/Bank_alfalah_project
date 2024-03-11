import logging
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone

import cv2
import face_recognition
import requests
from mtcnn.mtcnn import MTCNN

# Constants
FILENAME = "webcam"
FACE_RECOGNITION_TOLERANCE = 0.4
REAPPEARANCE_THRESHOLD = 1.0
ID_DISAPPEAR_THRESHOLD = 1.0
ENCODINGS_FILE = "embeddings.pkl"
LOG_FILENAME = f"{FILENAME}.log"
VIDEO_PATH = 0
# VIDEO_PATH = f'/home/zubair/Xloop/bafl/face_recognition/branch_manager/video/{FILENAME}.mp4'
OUTPUT_VIDEO_PATH = f"/home/zubair/Xloop/bafl/face_recognition/branch_manager/video/output/{FILENAME}_tol_{FACE_RECOGNITION_TOLERANCE}.avi"
FOURCC = cv2.VideoWriter_fourcc(*"XVID")
JWT_TOKEN = "eyJhbGciOiJIUzI1NJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU"
API_BASE_URL = "http://13.126.160.174:5000"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true",
}
# Set up logging
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("Branc Manager")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("logs/branch_manager_LIVE.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
# Initialize necessary variables
face_detector = MTCNN()
face_details_dict = {}
last_appearance_times = {}
prev_elapsed = {}
last_detected_times = {}


# Load the known faces and embeddings saved in the last file
try:
    with open(ENCODINGS_FILE, "rb") as file:
        data = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    logging.error(f"Failed to load encodings file: {e}")
    sys.exit(1)


def initialize_face_details(name, current_time):
    """
    Initializes the details of a new face when it is first detected.

    Args:
        face_id (str): The unique identifier for the face.
        start_frame (int): The frame number when the face was first detected.
        current_time (float): The current time when the face was first detected.
    """
    face_details_dict[name] = {
        "total_duration": 0,
        "last_detected_time": current_time,
        "time_stamp": datetime.now(timezone(timedelta(hours=5))).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + str("+05:00"),
        #    'server_id': None  # Initialize server_id to None
    }
    prev_elapsed[name] = 0
    last_detected_times[name] = current_time


def update_existing_face_details(name, current_time):
    """
    Updates the details of an existing face in the face_details_dict. The details include the last detected time,
    total duration of detection, and the start frame for the next calculation of elapsed time. It also updates the
    last appearance time for the detected person in the last_appearance_times dictionary.

    Args:
        name (str): The name of the detected person.
        current_frame (int): The current frame number in the video.
        fps (float): The frames per second of the video.
        current_time (float): The current time in the video.

    Raises:
        KeyError: If the name does not exist in the face_details_dict dictionary.
    """
    try:
        # face_details_dict[name]['last_detected_time'] = current_time
        # elapsed_frames = current_frame - face_details_dict[name]['start_frame']
        logging.info(f"detected_person: {name}")
        elapsed_time = current_time - face_details_dict[name]["last_detected_time"]
        prev_elapsed[name] = face_details_dict[name][
            "total_duration"
        ]  # Store current time of detected person as previous for next update
        logging.info(f"previous_elapsed_time: {round(prev_elapsed[name],2)}")
        face_details_dict[name]["total_duration"] += elapsed_time
        face_details_dict[name]["last_detected_time"] = current_time
        # Update the last appearance time for the Detected Person
        last_appearance_times[name] = current_time

    except KeyError as e:
        logging.error(f"KeyError encountered in update_existing_face_details: {e}")


# Function to update face details
def update_face_details(name, current_time):
    current_time = time.time()
    if name not in face_details_dict:
        initialize_face_details(name, current_time)
    else:
        update_existing_face_details(name, current_time)


# Helper function to send data to the server
def send_data_on_threshold(name):
    """
    Sends data to the server when the REAPPEARANCE_THRESHOLD condition is met.

    Args:
        name (str): The name of the detected person.
    """
    try:
        data = {
            "startTimestamp": face_details_dict[name]["time_stamp"],
            "exitTimestamp": datetime.now(timezone(timedelta(hours=5))).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            + str("+05:00"),
        }
        url = f"{API_BASE_URL}/branch-manager-tracking?managerId=4&branchId=1"
        logging.info(f"Branch Manager Request_payload: {data}")
        response = requests.post(url, json=data, headers=HEADERS)
        if response is not None:
            if response and response.status_code // 100 == 2:
                # Handle successful response
                logging.info("Request Successful\nResponse: %s", response.text)
            else:
                logging.info(
                    f"Request failed with status code {response.raise_for_status()}"
                )
        else:
            pass

    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        logging.error("Failed to get a response")
    except Exception as e:
        logging.error(f"An error occurred while sending data on threshold: {e}")


# Function to check and generate alerts
def check_and_generate_alerts():
    current_time = time.time()
    for name, last_appearance_time in last_appearance_times.items():
        # last_detected_time = details['last_detected_time']
        elapsed_time_since_last_detected = current_time - last_appearance_time
        if elapsed_time_since_last_detected > ID_DISAPPEAR_THRESHOLD:
            total_duration = face_details_dict[name]["total_duration"]
            # Print alert in red color
            logging.warning(
                f"\033[91mAlert for Detected Person: {name}, Total Duration: {round(total_duration, 2)} seconds\033[0m"
            )


def process_face(faces, rgb, frame, font):
    """
    Processes a detected face, performs face recognition and handles the match if found.

    Args:
        face (dict): A dictionary containing the bounding box of the face.
        rgb (array): The RGB image.
        current_frame (int): The current frame number.
        frame (array): The current frame.
        font (cv2.FONT): The font to use for text in the frame.
        fps (float): The frames per second of the video.
    """
    logger.debug("Processing frame in Branch Manager")
    for face in faces:
        x, y, w, h = face["box"]
        encoding = face_recognition.face_encodings(rgb, [(y, x + w, y + h, x)])[0]
        matches = face_recognition.compare_faces(
            data["encodings"], encoding, FACE_RECOGNITION_TOLERANCE
        )
        if True in matches:
            handle_match(matches, x, y, w, h, frame, font)


def handle_match(matches, x, y, w, h, frame, font):
    """
    Handles a match found by the face recognition. It updates the face details, logs the elapsed time,
    adds the elapsed time to the frame, and sends data if the time since last detected exceeds the threshold.

    Args:
        matches (list): A list of boolean values indicating the matches.
        x (int): The x-coordinate of the top-left corner of the bounding box.
        y (int): The y-coordinate of the top-left corner of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the bounding box.
        current_frame (int): The current frame number.
        frame (array): The current frame.
        font (cv2.FONT): The font to use for text in the frame.
        fps (float): The frames per second of the video.
    """
    matched_idxs = [i for (i, b) in enumerate(matches) if b]
    counts = {}
    for i in matched_idxs:
        name = data["names"][i]
        counts[name] = counts.get(name, 0) + 1
    name = max(counts, key=counts.get)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 4)
    cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)
    update_face_details(name, time.time())
    elapsed_time = face_details_dict[name]["total_duration"]
    logging.info(f"elapsed time: {round(elapsed_time,2)}")
    cv2.putText(
        frame,
        f"Time: {round(float(elapsed_time),2)}s",
        (x, y + h + 10),
        font,
        0.5,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Time: {round(float(elapsed_time),2)}s",
        (x, y + h + 10),
        font,
        0.5,
        (255, 255, 255),
        1,
    )
    time_since_last_detected = elapsed_time - prev_elapsed[name]
    logging.info(f"time difference: {round(time_since_last_detected,2)}")
    if time_since_last_detected > REAPPEARANCE_THRESHOLD:
        send_data_on_threshold(name)
        # Reset total_duration to 0 after sending data
        face_details_dict[name]["total_duration"] = 0


def main():
    """
    Main function to start the face detection and recognition process on a video file.
    """
    try:
        logging.info("Streaming started")
        video_capture = cv2.VideoCapture(VIDEO_PATH)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        logging.info(f"frame_rate: {fps}")
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(
            OUTPUT_VIDEO_PATH, FOURCC, fps, (frame_width, frame_height)
        )
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            font = cv2.FONT_HERSHEY_DUPLEX
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(rgb)
            # current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            # logging.info(f'current_frame_of_video: {current_frame}')
            logging.info(
                f"current_time_of_video: {datetime.fromtimestamp(time.time()).strftime('%I:%M:%S %p')}"
            )

            process_face(faces, rgb, frame, font)
            check_and_generate_alerts()
            out.write(frame)
            cv2.imshow(
                f"{FILENAME}_{FACE_RECOGNITION_TOLERANCE}_tolerance_{REAPPEARANCE_THRESHOLD}sec",
                frame,
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        line_number = sys.exc_info()[-1].tb_lineno
        logging.error(
            f"An error occurred in the main function at line number {line_number}: {e}"
        )
    finally:
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
