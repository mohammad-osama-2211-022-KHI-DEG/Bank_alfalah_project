import cv2
import logging
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import threading
import httpx
import requests
import queue

from atm_functionality_detection.main_atm import process_frame, complainbox_telephone, Time, data_preparation, Alert
from suspecious_activity.main_suspecious import process_frame_thread, detect_suspicious_conditions, check_person_duration, load_model
from guard.main_guard import process_guard_frame_thread
#from atm_functionality_detection.main import *

# For video
VIDEO_PATH = '../team_raqim_rehman/atm_functionality_detection/ATM_working.mp4'


# For Streaming
# username = 'usama.xloop'
# password = 'Xloop@123'
# ip_address = '192.168.6.19'
# stream_url = f"rtsp://{username}:{password}@{ip_address}:554/cam/realmonitor?channel=1&subtype=1"

model_path_atm_url = '../team_raqim_rehman/atm_functionality_detection/new_best.pt'
model_suspecious_url = '../team_raqim_rehman/suspecious_activity/suspecious_updated.pt'
model_guard_detection_url = '../team_raqim_rehman/guard/guard_attire.pt'
model_guard_features_url = '../team_raqim_rehman/guard/guard_features.pt'

model_suspecious = YOLO(model_suspecious_url)
model_atm = YOLO(model_path_atm_url)
model_guard_detection = YOLO(model_guard_detection_url)
model_guard_features = YOLO(model_guard_features_url)


LOG_FILENAME = '../team_raqim_rehman/thread_log.log'
logging.basicConfig(
    filename=LOG_FILENAME,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define HTTP headers and URL
headers = {
    "Content-Type": "application/json",
    "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtdWhhbW1hZG9zYW1hLmhxQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwNjYwMDA3NywiZXhwIjoxNzA2OTAwMDc3fQ.4zBF4ZpY_nXax4IhBrdHLQqYw2yj31p5-ZxlVCW5D68",
    "X-XSRF-TOKEN": "693cb460-a9ed-4d04-95c5-71bb46f931e7",
    "X-SERVER-TO-SERVER" : "true"
}
url = 'http://13.235.71.140:5000/atm-functionality'

# Initialize variables
atm_detected = False
total_working_count = 0
total_notworking_count = 0
person = False
counter = 0
count = 0

MAX_QUEUE_SIZE = 100  # Define your maximum queue size
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def main():
    """
    Main function to start the face detection and recognition process on a video file.
    """
    # Open the video capture
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    logging.info(f"frame_rate: {fps}")

    # Target processing interval for 10 fps
    FPS_INTERVEL = 10
    frame_count = 0

    try:
        logging.info("Streaming started")
        print("Streaming started")

        # Loop through the video frames
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % FPS_INTERVEL != 0:
                logging.info(f"frame : {frame_count % FPS_INTERVEL}")
                continue
            # Put the frame into the queue
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                logging.warning("Frame queue is full, dropping frames.")
            process_frames()

    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
    finally:
        # Release the video capture
        video_capture.release()

# Function to process frames
def process_frames():
    first_person_time = None
    guard_attire_statuses = []
    previous_guard_attire_statuses = []
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Process the frame
            t1 = threading.Thread(target=process_frame, args=(frame, model_atm, atm_detected, total_working_count, total_notworking_count, counter, url, headers))
            t2 = threading.Thread(target=process_frame_thread, args=(model_suspecious, frame, first_person_time))
            t3 = threading.Thread(target=process_guard_frame_thread, args=(model_guard_detection, model_guard_features, frame, guard_attire_statuses, previous_guard_attire_statuses))
            t1.start()
            t2.start()
            t3.start()
            t1.join()
            t2.join()
            t3.join()
        else:
            break



if __name__ == '__main__':
    # Start the frame processing thread
    main()
