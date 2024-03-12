import cv2
import datetime
from ultralytics import YOLO
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def setup_logging(log_filename):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def detect_suspicious_conditions(class_ids):
    """
    Detect suspicious conditions based on detected class IDs.
    """
    suspicious_conditions = []
    if 4 in class_ids:
        suspicious_conditions.append('Helmet detected')
    num_persons = (class_ids == 5).sum()
    if num_persons > 2:
        suspicious_conditions.append('More than 2 persons detected')
    return suspicious_conditions

def check_person_duration(num_persons, first_person_time):
    """
    Check if one person is standing for more than 5 minutes.
    """
    if num_persons == 1 and first_person_time is None:
        return datetime.datetime.now()
    elif num_persons != 1:
        return None
    else:
        elapsed_time = datetime.datetime.now() - first_person_time
        if elapsed_time.total_seconds() > 300:  # 5 minutes in seconds
            return None
        return first_person_time


def process_frame_thread(MODEL_PATH, frame, first_person_time):
    logger.info("!!! Suspicious activity search in progress !!! ")
    # Assuming MODEL_PATH is a function or an object that takes frame as input and returns results
    results = MODEL_PATH(frame)

    # Get class IDs from the detected boxes
    class_ids = results[0].boxes.cls.numpy()

    # Detect suspicious conditions
    suspicious_conditions = detect_suspicious_conditions(class_ids)
    for condition in suspicious_conditions:
        print('Suspicious:', condition)
        logger.info(f"Suspicious: {condition}")

    # Check if one person is standing for more than 5 minutes
    first_person_time = check_person_duration((class_ids == 5).sum(), first_person_time)
    logger.info(f"First person standing time: {first_person_time} seconds")

    return first_person_time

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Constants
    LOG_FILENAME = os.getenv("LOG_FILENAME")
    VIDEO_PATH = os.getenv("VIDEO_PATH")
    MODEL_PATH = os.getenv("MODEL_PATH")

    # Set up logging
    setup_logging(LOG_FILENAME)

    # Load the YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Get current FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Current FPS:", fps)

    # Initialize variables for tracking suspicious activity
    first_person_time = None
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Skip frames if not the 10th frame
        if frame_count % 10 != 0:
            logging.info(f"10th frame : {frame_count % 10}")
            continue

        frame = cv2.resize(frame, (640, 480))

        results = model(frame)

        # Get class IDs from the detected boxes
        class_ids = results[0].boxes.cls.numpy()

        # Detect suspicious conditions
        suspicious_conditions = detect_suspicious_conditions(class_ids)
        for condition in suspicious_conditions:
            print('Suspicious:', condition)
            logging.info(f"Suspicious: {condition}")

        # Check if one person is standing for more than 5 minutes
        first_person_time = check_person_duration((class_ids == 5).sum(), first_person_time)

        # Display the processed frame
        cv2.imshow('Branch', results[0].plot())
        cv2.imshow('Branch_frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
