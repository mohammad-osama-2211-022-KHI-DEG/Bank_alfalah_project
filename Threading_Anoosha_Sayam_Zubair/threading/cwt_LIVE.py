import cv2
from mtcnn import MTCNN
import face_recognition
import time
import logging
import sys
import requests
from datetime import datetime, timezone, timedelta

# FILENAME = "web_cam"
FILENAME = "CWT_LIVE_LEAP"
CONFIDENCE_THRESHOLD = 0.97
FACE_RECOGNITION_TOLERANCE = 0.50
TIME_THRESHOLD = 1.0
ID_DISAPPEAR_THRESHOLD = 1.0
NUM_JITTERS = 10
# LOG_FILENAME = f"../team_anoosha/Threading/logs/{FILENAME}_t1.log"
# VIDEO_PATH = 2
VIDEO_PATH = f'../customer_waiting_time/video/LEAP_1.avi'
OUTPUT_PATH = f'../customer_waiting_time/video/output/{FILENAME}_tol_{FACE_RECOGNITION_TOLERANCE}_{TIME_THRESHOLD}sec.avi'
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU"
API_BASE_URL = 'http://13.126.160.174:5000'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true"
}

# Set up logging
logging.basicConfig(
    filename=f"/home/zubair/Xloop/bafl/face_recognition/customer_waiting_time/logs/{FILENAME}_t1.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)    
# Setup logging
logger = logging.getLogger('CWT_LIVE')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/cwt_LIVE.log', mode= 'w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


face_detector = MTCNN()
face_encodings_dict = {}
face_details_dict = {}
last_appearance_times = {}
last_detected_times = {}
current_face_id = 0
prev_elapsed = {}
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
def initialize_face_details(face_id, current_time):
    """
    Initializes the details of a new face when it is first detected.

    Args:
        face_id (str): The unique identifier for the face.
        start_frame (int): The frame number when the face was first detected.
        current_time (float): The current time when the face was first detected.
    """
    face_details_dict[face_id] = {
                                  'total_duration': 0, 
                                  'last_detected_time': current_time,
                                  'time_stamp': datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00'),
                                  'notification_sent': False,
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
    logging.info(f"New Face Request_Payload: {data}")
    response = send_server_request('/customer-wait-time', data)
    if response is not None:
        if response and response.status_code // 100 == 2:
            returned_id = response.json().get('id')
            face_details_dict[face_id]['server_id'] = returned_id
            logging.info(f"Request Successful\nResponse: {response.text}")
        else:
            logging.info(f"Request failed with status code {response.status_code}")
    else:
        logging.error("Failed to get a response")

def update_existing_face_details(face_id, current_time):
    """
    Updates the details of an existing face with the elapsed time since the last detection.

    Args:
        face_id (str): The unique identifier for the face.
        current_frame (int): The current frame number.
        fps (float): The frames per second of the video.
        current_time (float): The current time of the video.
    """
    try:
        logging.info(f'face id: {face_id}')
        elapsed_time = current_time - face_details_dict[face_id]['last_detected_time']
        prev_elapsed[face_id] = face_details_dict[face_id]['total_duration']  # Store current as previous for next update
        logging.info(f'prev elapsed time: {round(prev_elapsed[face_id], 2)}')
        face_details_dict[face_id]['total_duration'] += elapsed_time
        face_details_dict[face_id]['last_detected_time'] = current_time
        last_appearance_times[face_id] = current_time
    except KeyError as e:
        logging.error(f"KeyError encountered in update_existing_face_details: {e}")

def update_face_details(face_id, current_time):
    current_time = time.time()
    if face_id not in face_details_dict:
        initialize_face_details(face_id, current_time)
    else:
        update_existing_face_details(face_id, current_time)
    logging.info(f"total_duration: {face_details_dict}")

def check_and_generate_alerts():
    current_time = time.time()
    for face_id, last_appearance_time in last_appearance_times.items():
        elapsed_time_since_last_detected = current_time - last_appearance_time
        server_id = face_details_dict[face_id].get('server_id')  # Extract server_id from face_details_dict

        if elapsed_time_since_last_detected > ID_DISAPPEAR_THRESHOLD and server_id is not None:
            total_duration = face_details_dict[face_id]['total_duration']
            data = {
                "country": "pakistan",
                "branch": "clifton",
                "city": "karachi",
                "timestamp": datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00'), 
                "status": "EXIT"
            }
            endpoint = f'/customer-wait-time/exit/{server_id}'
            logging.info(f"Dissappear Face_id_{server_id} Request_Payload: {data}")
            response = send_server_request(endpoint, data, method='put')
            if response is not None:
                if response and response.status_code // 100 == 2:
                    # Handle successful response
                    logging.info(f"Request Successful\nResponse: {response.text}")
                else:
                    logging.info(f"Request failed with status code {response.status_code}")
            else:
                logging.error("Failed to get a response")
            # Log the alert message
            logging.warning(f"\033[91mAlert for Face ID: {face_id}, Total Duration: {round(total_duration, 2)} seconds\033[0m")

def process_faces(faces, frame, rgb):

    logger.debug('Processing frame in CWT')
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']
        if confidence > CONFIDENCE_THRESHOLD:
            encodings = face_recognition.face_encodings(rgb, [(y, x + width, y + height, x)], num_jitters=NUM_JITTERS)
            if len(encodings) > 0:
                process_encodings(encodings, frame, x, y, width, height)

def process_encodings(encodings, frame, x, y, width, height):
    encoding = encodings[0]
    matches = face_recognition.compare_faces(list(face_encodings_dict.values()), encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
    if any(matches):
        process_matched_face(matches, frame, x, y, width, height)
    else:
        process_unmatched_face(encoding, frame, x, y, width, height)

def process_matched_face(matches, frame, x, y, width, height):
    match_index = matches.index(True)
    recognized_person_id = list(face_encodings_dict.keys())[match_index]
    draw_recognized_face(frame, x, y, width, height, recognized_person_id)
    update_face_details(recognized_person_id, time.time())
    display_elapsed_time(frame, x, y, width, height, recognized_person_id)
    reset_total_duration_if_needed(recognized_person_id)
    face_id = recognized_person_id
    if face_details_dict[face_id]['total_duration'] > TIME_THRESHOLD and not face_details_dict[face_id]['notification_sent']:
        data = {
            "timestamp": datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00'),
            "message": f"Id: {face_id} is waiting for {round(face_details_dict[face_id]['total_duration'], 2)} seconds",
            "country": "pakistan",
            "city": "karachi",
            "branch": "clifton",
            "usecase": "customer waiting time",
        }
        logging.info(f"Notification Request_payload:{data}")
        face_details_dict[face_id]['notification_sent'] = True
        response = send_server_request('/notification', data)
        if response is not None:
            if response and response.status_code // 100 == 2:
                    # Handle successful response
                    logging.info(f"Request Successful\nResponse: {response.text}")
                    face_details_dict[face_id]['notification_sent'] = True 
            else:
                    logging.info(f"Request failed with status code {response.status_code}")
        else:
            logging.error("Failed to get a response")

def process_unmatched_face(encoding, frame, x, y, width, height):
    new_face_id = generate_unique_id()
    draw_unrecognized_face(frame, x, y, width, height, new_face_id)
    update_face_encodings(encoding, new_face_id)

def draw_recognized_face(frame, x, y, width, height, face_id):
    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 100, 255), 4)
    cv2.putText(frame, f'customer_{face_id}', (x + 6, y - 6), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, f'customer_{face_id}', (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

def draw_unrecognized_face(frame, x, y, width, height, face_id):
    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
    cv2.putText(frame, f'customer_{face_id}', (x + 6, y + height - 6), font, 0.5, (255, 255, 255), 1)

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
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        logging.info(f'frame_rate: {fps}')
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(OUTPUT_PATH, FOURCC, fps, (frame_width, frame_height))
        # frame_skip_count = 0
        # skip_interval = 5
        while True:

            ret, frame = video_capture.read()
            if not ret:
                break   
            # frame_skip_count +=1
            # if frame_skip_count % skip_interval != 0:
            #     continue
            faces = face_detector.detect_faces(frame)
            current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            logging.info(f'current_frame_of_video: {current_frame}')
            logging.info(f"current_time_of_video: {datetime.fromtimestamp(time.time()).strftime('%I:%M:%S %p')}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            process_faces(faces, frame, rgb)
            check_and_generate_alerts()
            
            out.write(frame)
            cv2.imshow(f'{FILENAME}_{FACE_RECOGNITION_TOLERANCE}_tolerance_{TIME_THRESHOLD}sec', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()
    except Exception as e:
        line_number = sys.exc_info()[-1].tb_lineno
        logging.error(f"An error occurred in the main function at line number {line_number}: {e}")
        
if __name__ == "__main__":
    main()