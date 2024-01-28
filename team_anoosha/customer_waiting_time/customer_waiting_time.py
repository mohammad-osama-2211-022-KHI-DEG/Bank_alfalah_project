import cv2
from mtcnn import MTCNN
import face_recognition
import time
import logging
import requests
from datetime import datetime, timezone, timedelta

logging.basicConfig(filename='output/logs/mart.log', filemode= 'w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CONFIDENCE_THRESHOLD = 0.95
FACE_RECOGNITION_TOLERANCE = 0.55
TIME_THRESHOLD = 1.0
ID_DISAPPEAR_THRESHOLD = 0.2
VIDEO_PATH = 'video/mart.mp4'
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
OUTPUT_PATH = 'output/video_output/mart.avi'
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtdWhhbW1hZG9zYW1hLmhxQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwNjM0MzI0OCwiZXhwIjoxNzA2NjQzMjQ4fQ.itpLlCgVvdJX5wbx76sR6Nvz87YcnFUDl5Gs0DsOCUA"
API_BASE_URL = 'http://13.235.71.140:5000'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true"
}
face_detector = MTCNN()
face_encodings_dict = {}
face_details_dict = {}
last_appearance_times = {}
last_detected_times = {}
current_face_id = 0
prev_elapsed = {}

# Define the font variable globally if it's used in multiple functions
font = cv2.FONT_HERSHEY_DUPLEX

# Define the missing functions and variables
def generate_unique_id():
    """
    Generates a unique identifier for a new face.

    Returns:
        str: A string representation of the current face ID.
    """
    global current_face_id
    current_face_id += 1
    return str(current_face_id)

def update_face_encodings(encoding, face_id):
    """
    Updates the face encodings dictionary with the new encoding for the specified face ID.

    Args:
        encoding: The face encoding to be stored.
        face_id (str): The unique identifier for the face.
    """
    face_encodings_dict[face_id] = encoding

# Define the font variable globally if it's used in multiple functions
font = cv2.FONT_HERSHEY_DUPLEX

# Helper function to send data to the server
def send_server_request(endpoint: str, data: dict, method: str = 'post') -> requests.Response:
    """
    Sends a request to the server with the given data.

    Args:
        endpoint (str): The API endpoint to send the request to.
        data (dict): The data payload for the request.
        method (str): The HTTP method to use (default is 'post').

    Returns:
        requests.Response: The response from the server.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.lower() == 'post':
            response = requests.post(url, json=data, headers=HEADERS)
        elif method.lower() == 'put':
            response = requests.put(url, json=data, headers=HEADERS)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the request: {e}")
        return None
    
# Update the functions that require `current_time`
def initialize_face_details(face_id, start_frame, current_time):
    """
    Initializes the details of a new face when it is first detected.

    Args:
        face_id (str): The unique identifier for the face.
        start_frame (int): The frame number when the face was first detected.
        current_time (float): The current time when the face was first detected.
    """
    face_details_dict[face_id] = {'start_frame': start_frame, 
                                  'total_duration': 0, 
                                  'last_detected_time': current_time,
                                  'time_stamp': datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S%z'),
                                  'server_id': None  # Initialize server_id to None                                 
                                }
    prev_elapsed[face_id] = 0
    last_detected_times[face_id] = current_time
    data = {
        "country": "pakistan",
        "branch": "clifton",
        "city": "karachi",
        "timestamp": face_details_dict[face_id]['time_stamp'],
        "status": "PRESENT"
    }
    response = send_server_request('/customer-wait-time', data)
    if response and response.status_code // 100 == 2:
        returned_id = response.json().get('id')
        face_details_dict[face_id]['server_id'] = returned_id
        logging.info("Request Successful\nResponse: %s", response.text)
    else:
        logging.info(f"Request failed with status code {response.status_code}")

def update_existing_face_details(face_id, current_frame, fps, current_time):
    """
    Updates the details of an existing face with the elapsed time since the last detection.

    Args:
        face_id (str): The unique identifier for the face.
        current_frame (int): The current frame number.
        fps (float): The frames per second of the video.
        current_time (float): The current time of the video.
    """
    try:
        face_details_dict[face_id]['last_detected_time'] = current_time 
        elapsed_frames = current_frame - face_details_dict[face_id]['start_frame']
        logging.info(f'face id: {face_id}')
        elapsed_time = elapsed_frames / fps
        prev_elapsed[face_id] = face_details_dict[face_id]['total_duration']  # Store current as previous for next update
        logging.info(f'prev elapsed time: {round(prev_elapsed[face_id], 2)}')
        face_details_dict[face_id]['total_duration'] += elapsed_time
        face_details_dict[face_id]['start_frame'] = current_frame
        last_appearance_times[face_id] = time.time()
    except KeyError as e:
        logging.error(f"KeyError encountered in update_existing_face_details: {e}")

def update_face_details(face_id, start_frame, fps, current_time):
    if face_id not in face_details_dict:
        initialize_face_details(face_id, start_frame, current_time)
    else:
        update_existing_face_details(face_id, start_frame, fps, current_time)

def check_and_generate_alerts(current_time):
    for face_id, details in face_details_dict.items():
        last_detected_time = details['last_detected_time']
        elapsed_time_since_last_detected = current_time - last_detected_time
        total_duration = details['total_duration']
        server_id = details.get('server_id')  # Extract server_id from face_details_dict

        if elapsed_time_since_last_detected > ID_DISAPPEAR_THRESHOLD and server_id is not None:
            timestamp = datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S%z')
            data = {
                "country": "pakistan",
                "branch": "clifton",
                "city": "karachi",
                "timestamp": timestamp, 
                "status": "EXIT"
            }
            endpoint = f'/customer-wait-time/exit/{server_id}'
            response = send_server_request(endpoint, data, method='put')
            if response and response.status_code // 100 == 2:
                # Handle successful response
                logging.info("Request Successful\nResponse: %s", response.text)
            else:
                logging.info(f"Request failed with status code {response.status_code}")
            # Log the alert message
            logging.warning(f"\033[91mAlert for Face ID: {face_id}, Total Duration: {round(total_duration, 2)} seconds, Exit: {datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S%Z')}\033[0m")

def process_faces(faces, frame, rgb, current_frame, fps, current_time):
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']
        if confidence > CONFIDENCE_THRESHOLD:
            encodings = face_recognition.face_encodings(rgb, [(y, x + width, y + height, x)], num_jitters=20)
            if len(encodings) > 0:
                process_encodings(encodings, frame, x, y, width, height, current_frame, fps)

def process_encodings(encodings, frame, x, y, width, height, current_frame, fps):
    encoding = encodings[0]
    matches = face_recognition.compare_faces(list(face_encodings_dict.values()), encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
    if any(matches):
        process_matched_face(matches, frame, x, y, width, height, current_frame, fps)
    else:
        process_unmatched_face(encoding, frame, x, y, width, height)

def process_matched_face(matches, frame, x, y, width, height, current_frame, fps):
    match_index = matches.index(True)
    recognized_person_id = list(face_encodings_dict.keys())[match_index]
    draw_recognized_face(frame, x, y, width, height, recognized_person_id)
    update_face_details(recognized_person_id,  current_frame, fps, current_time = current_frame/fps)
    display_elapsed_time(frame, x, y, width, height, recognized_person_id)
    reset_total_duration_if_needed(recognized_person_id)

def process_unmatched_face(encoding, frame, x, y, width, height):
    new_face_id = generate_unique_id()
    draw_unrecognized_face(frame, x, y, width, height, new_face_id)
    update_face_encodings(encoding, new_face_id)

def draw_recognized_face(frame, x, y, width, height, face_id):
    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 100, 255), 4)
    cv2.putText(frame, face_id, (x + 6, y - 6), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, face_id, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

def draw_unrecognized_face(frame, x, y, width, height, face_id):
    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
    cv2.putText(frame, face_id, (x + 6, y + height - 6), font, 0.5, (255, 255, 255), 1)

def display_elapsed_time(frame, x, y, width, height, face_id):
    elapsed_time = face_details_dict[face_id]['total_duration']
    cv2.putText(frame, f"Time: {round(float(elapsed_time),2)}s", (x, y + height+10), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {round(float(elapsed_time),2)}s", (x, y + height+10), font, 0.5, (255, 255, 255), 1)

def reset_total_duration_if_needed(face_id):
    elapsed_time = face_details_dict[face_id]['total_duration']
    logging.info(f'elapsed time: {round(elapsed_time,2)}') 
    time_since_last_detected = elapsed_time - prev_elapsed[face_id]
    logging.info(f'time difference: {round(time_since_last_detected,2)}')
    if time_since_last_detected > TIME_THRESHOLD:
        face_details_dict[face_id]['total_duration'] = 0

# Main function
def main():
    """
    Main function to start the face detection and recognition process on a video file.
    """
    try:
        video_capture = cv2.VideoCapture(VIDEO_PATH)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        logging.info(f'frame_rate: {fps}')
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(OUTPUT_PATH, FOURCC, fps, (frame_width, frame_height))
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            font = cv2.FONT_HERSHEY_DUPLEX
            faces = face_detector.detect_faces(frame)
            current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            logging.info(f'current_frame_of_video: {current_frame}')
            current_time = current_frame/fps
            logging.info(f'current_time_of_video: {round(current_time, 2)}')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            process_faces(faces, frame, rgb, current_frame, fps, current_time)
            out.write(frame)
            cv2.imshow('Video', frame)
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