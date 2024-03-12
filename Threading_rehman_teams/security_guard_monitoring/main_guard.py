import cv2
from datetime import datetime, timezone, timedelta
from ultralytics import YOLO
import os
import requests
from requests.auth import HTTPBasicAuth

global jwt_token


def get_jwt_token():
    """
    Function to get JWT token from the authentication endpoint
    """

    headers = {
        "X-SERVER-TO-SERVER": "true"
    }

    auth_url = "http://13.126.160.174:5000/auth/user"
    username = "aniqamasood111@gmail.com"
    password = "Aniqa123"

    # Make a request to the authentication endpoint with basic authentication
    response = requests.get(auth_url, auth=HTTPBasicAuth(username, password), headers=headers)

    # Check if the request was successful (status code 2xx)
    if response.status_code // 100 == 2:
        # Extract and return the JWT token
        return response.headers['Authorization']
    else:
        # Handle authentication error
        raise ValueError(f"Authentication failed with status code {response.status_code}")
    
    
def send_data_to_endpoint(data, endpoint, jwt_token):
#     """
#     Function to send data to the specified endpoint with JWT token
    # """

    jwt = jwt_token

    headers = {
        "Content-Type": "application/json",
        "Authorization": jwt,
        "X-SERVER-TO-SERVER": "true"
    }

    # Make a request to the specified endpoint with the provided data and headers
    response = requests.post(endpoint, json=data, headers=headers)

    # Check if the request was successful (status code 2xx)
    if response.status_code // 100 == 2:
        print("Data sent successfully.")
    elif response.status_code == 401:
        # If unauthorized, refresh the JWT token and retry
        print("JWT token expired. Refreshing and retrying...")
        jwt_token = get_jwt_token()
        #jwt_token = new_jwt_token
        send_data_to_endpoint(data, endpoint, jwt_token)

    else:
        # Handle other errors
        print(f"Error sending data. Status code: {response.status_code}, Response: {response.text}")


def load_model(model_path):
    """
    Load the YOLOv8 model.
    """
    return YOLO(model_path)

def process_guard_frame_thread(model_guard_detection, model_guard_features, frame, guard_attire_statuses, previous_guard_attire_statuses):
    # model_guard_detection = load_model(DETECTION_MODEL)
    # model_guard_features = load_model(FEATURES_MODEL)

    results_guard_detection = model_guard_detection(frame)

    for guard_index, guard_box in enumerate(results_guard_detection[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, guard_box.tolist())
            cropped_guard = frame[y1:y2, x1:x2]

            # Check attire status of each guard using the second YOLO model
            results_guard_features = model_guard_features(cropped_guard)

            # Initialize the uniform status for the current guard
            current_uniform_state = 'IMPROPER'

            # Check if cap and shoes are present in the attire of the current guard
            class_ids = results_guard_features[0].boxes.cls.numpy()
            if 0 in class_ids and 1 in class_ids:
                current_uniform_state = 'PROPER'

            # Append the uniform status of the current guard to the list
            guard_attire_statuses.append(current_uniform_state)

    if guard_attire_statuses != previous_guard_attire_statuses:

        # Send data to the endpoint
        timestamp = datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
        data_to_push = {
            'uniformStatus': guard_attire_statuses,
            'timestamp': timestamp
        }


        # Send data to the endpoint
        #send_data_to_endpoint(data_to_push, target_endpoint, jwt_token)

        #response = requests.post('http://13.233.56.158:5000/security-guard-tracking2?branchId=1', json=data_to_push)
        #print(response.text)
        #print(data_to_push)

    # Update previous status for the next iteration
    previous_guard_attire_statuses = guard_attire_statuses.copy()

    # Reset guard_attire_statuses for the next frame
    guard_attire_statuses = []




def main():
    # Get initial JWT token
    jwt_token = get_jwt_token()
    print(jwt_token)

    # Endpoint for sending data
    target_endpoint = "http://13.126.160.174:5000/security-guard-tracking2?branchId=1"


    # Load the YOLOv8 models
    model_guard_detection = YOLO('./guard_attire.pt')
    model_guard_features = YOLO('./guard_features.pt')

    # Load video
    video_path = './2_guards.mp4'
    cap = cv2.VideoCapture(video_path)


    # Initialize variables
    guard_attire_statuses = []
    previous_guard_attire_statuses = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Detect guards in the frame using the first YOLO model
        results_guard_detection = model_guard_detection(frame)

        for guard_index, guard_box in enumerate(results_guard_detection[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, guard_box.tolist())
            cropped_guard = frame[y1:y2, x1:x2]

            # Check attire status of each guard using the second YOLO model
            results_guard_features = model_guard_features(cropped_guard)

            # Initialize the uniform status for the current guard
            current_uniform_state = 'IMPROPER'

            # Check if cap and shoes are present in the attire of the current guard
            class_ids = results_guard_features[0].boxes.cls.numpy()
            if 0 in class_ids and 1 in class_ids:
                current_uniform_state = 'PROPER'

            # Append the uniform status of the current guard to the list
            guard_attire_statuses.append(current_uniform_state)

            cv2.imshow(f"Results {guard_index+1}", results_guard_features[0].plot())
            cv2.imshow(f"Object {guard_index+1}", cropped_guard)

        # Check if the status has changed
        if guard_attire_statuses != previous_guard_attire_statuses:

            # Send data to the endpoint
            timestamp = datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
            data_to_push = {
                'uniformStatus': guard_attire_statuses,
                'timestamp': timestamp
            }


            # Send data to the endpoint
            send_data_to_endpoint(data_to_push, target_endpoint, jwt_token)

            #response = requests.post('http://13.233.56.158:5000/security-guard-tracking2?branchId=1', json=data_to_push)
            #print(response.text)
            print(data_to_push)

        # Update previous status for the next iteration
        previous_guard_attire_statuses = guard_attire_statuses.copy()

        # Reset guard_attire_statuses for the next frame
        guard_attire_statuses = []

        cv2.imshow('Jinnah Avenue, Islamabad', results_guard_detection[0].plot())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

