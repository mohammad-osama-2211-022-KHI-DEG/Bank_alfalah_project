import cv2
from mtcnn.mtcnn import MTCNN
import face_recognition
import time
import logging
import sys
import pickle
import requests
from datetime import datetime, timezone, timedelta

# Set up logging
logging.basicConfig(filename='output/logs/nov.log', filemode= 'w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
FACE_RECOGNITION_TOLERANCE = 0.4
REAPPEARANCE_THRESHOLD = 1.0
ID_DISAPPEAR_THRESHOLD = 1.0
ENCODINGS_FILE = 'encodings/face_enc_all_cnn_big.pkl'
VIDEO_PATH = 'video/nov.mp4'
OUTPUT_VIDEO_PATH = 'output/video_output/nov.avi'
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtdWhhbW1hZG9zYW1hLmhxQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwNjM0MzI0OCwiZXhwIjoxNzA2NjQzMjQ4fQ.itpLlCgVvdJX5wbx76sR6Nvz87YcnFUDl5Gs0DsOCUA"
API_BASE_URL = 'http://13.235.71.140:5000'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true"
}

# Initialize necessary variables
face_detector = MTCNN()
face_details_dict = {}
last_appearance_times = {}
prev_elapsed = {}
last_detected_times = {}


# Load the known faces and embeddings saved in the last file
try:
    with open(ENCODINGS_FILE, 'rb') as file:
        data = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    logging.error(f"Failed to load encodings file: {e}")
    sys.exit(1)

def initialize_face_details(name, start_frame, current_time):
    """
    Initializes the details of a new face when it is first detected.

    Args:
        face_id (str): The unique identifier for the face.
        start_frame (int): The frame number when the face was first detected.
        current_time (float): The current time when the face was first detected.
    """
    face_details_dict[name] = {'start_frame': start_frame, 
                               'total_duration': 0,
                               'last_detected_time': current_time,
                               'time_stamp': datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S%z'),
                            #    'server_id': None  # Initialize server_id to None                                 
                              }
    prev_elapsed[name] = 0
    last_detected_times[name] = current_time

def update_existing_face_details(name, current_frame, fps, current_time):
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
        face_details_dict[name]['last_detected_time'] = current_time
        elapsed_frames = current_frame - face_details_dict[name]['start_frame']
        logging.info(f'detected_person: {name}')
        elapsed_time = elapsed_frames / fps
        prev_elapsed[name] = face_details_dict[name]['total_duration']  # Store current time of detected person as previous for next update
        logging.info(f'previous_elapsed_time: {round(prev_elapsed[name],2)}')
        face_details_dict[name]['total_duration'] += elapsed_time
        face_details_dict[name]['start_frame'] = current_frame
        # Update the last appearance time for the Detected Person
        last_appearance_times[name] = time.time()

    except KeyError as e:
        logging.error(f"KeyError encountered in update_existing_face_details: {e}")

# Function to update face details
def update_face_details(name, start_frame, fps, current_time):
    if name not in face_details_dict:
        initialize_face_details(name, start_frame, current_time)
    else:
        update_existing_face_details(name, start_frame, fps, current_time)

class CustomRequestException(Exception):
    """Custom exception for errors in server requests."""
    pass

# Helper function to send data to the server
def send_data_on_threshold(name):
    """
    Sends data to the server when the REAPPEARANCE_THRESHOLD condition is met.

    Args:
        name (str): The name of the detected person.
    """
    try:
        timestamp = datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S%z')
        data = {
            "country": "pakistan",
            "branch": "clifton",
            "city": "karachi",
            "timestamp": face_details_dict[name]['time_stamp'], 
            "branchManagerName": name,
            "startTimestamp": face_details_dict[name]['time_stamp'],
            "exitTimestamp": timestamp,
            "areaManagerVisits": 0
        }
        
        url = f"{API_BASE_URL}/branch-manager"
        response = requests.post(url, json=data, headers=HEADERS)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

        if response and response.status_code // 100 == 2:
            # Handle successful response
            logging.info("Request Successful\nResponse: %s", response.text)
        else:
            logging.info(f"Request failed with status code {response.status_code}")

        # Reset total_duration to 0 after sending data
        face_details_dict[name]['total_duration'] = 0
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        raise CustomRequestException(f"Error sending request to {url}: {e}") from e
    except Exception as e:
        logging.error(f"An error occurred while sending data on threshold: {e}")

# Function to check and generate alerts
def check_and_generate_alerts(current_time):
    for name, details in face_details_dict.items():
        last_detected_time = details['last_detected_time']
        elapsed_time_since_last_detected = current_time - last_detected_time
        if elapsed_time_since_last_detected > REAPPEARANCE_THRESHOLD:
            total_duration = details['total_duration']
            # Print alert in red color
            logging.warning(f"\033[91mAlert for Detected Person: {name}, Total Duration: {round(total_duration, 2)} seconds, Exit: {datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S%Z')}\033[0m")

def process_face(face, rgb, current_frame, frame, font, fps):
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
    x, y, w, h = face['box']
    encoding = face_recognition.face_encodings(rgb, [(y, x + w, y + h, x)])[0]
    matches = face_recognition.compare_faces(data["encodings"], encoding, FACE_RECOGNITION_TOLERANCE)
    if True in matches:
        handle_match(matches, x, y, w, h, current_frame, frame, font, fps)

def handle_match(matches, x, y, w, h, current_frame, frame, font, fps):
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
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 100, 255), 4)
    cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)
    update_face_details(name, current_frame, fps, current_time = current_frame/fps)
    elapsed_time = face_details_dict[name]['total_duration']
    logging.info(f'elapsed time: {round(elapsed_time,2)}')
    cv2.putText(frame, f"Time: {round(float(elapsed_time),2)}s", (x, y + h+10), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {round(float(elapsed_time),2)}s", (x, y + h+10), font, 0.5, (255, 255, 255), 1)
    time_since_last_detected = elapsed_time - prev_elapsed[name]
    logging.info(f'time difference: {round(time_since_last_detected,2)}')
    if time_since_last_detected > REAPPEARANCE_THRESHOLD:
        send_data_on_threshold(name)

def main():
    """
    Main function to start the face detection and recognition process on a video file.
    """
    try:
        logging.info("Streaming started")
        video_capture = cv2.VideoCapture(VIDEO_PATH)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        logging.info(f'frame_rate: {fps}')
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, FOURCC, fps, (frame_width, frame_height))
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            font = cv2.FONT_HERSHEY_DUPLEX
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(rgb)
            current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            logging.info(f'current_frame_of_video: {current_frame}')
            current_time = current_frame/fps
            logging.info(f'current_time_of_video: {round(current_time,2)}')
            for face in faces:
                process_face(face, rgb, current_frame, frame, font, fps)
            out.write(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            check_and_generate_alerts(current_time)
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
    finally:
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()