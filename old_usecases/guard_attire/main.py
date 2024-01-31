import cv2
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import numpy as np

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/home/xloop/Bank_Alfalah/wait_time_tracking/silent-space-358606-firebase-adminsdk-j9k7s-6bfcfa7775.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the YOLOv8 model
model = YOLO('/home/xloop/Bank_Alfalah/Bank_alfalah_project/guard_attire/best.pt')

# Load video
video_path = '/home/xloop/Bank_Alfalah/Bank_alfalah_project/guard_attire/WhatsApp Video 2023-10-30 at 8.04.39 PM.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize variables
guard_present = False
class_present = False
class_start_time = None
previous_guard_present = False
previous_cap_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for faster processing (adjust resolution as needed)
    frame = cv2.resize(frame, (640, 480))

    results = model(frame)
    
    # Get class IDs from the detected boxes
    class_ids = results[0].boxes.cls.numpy()

    # Check if "uniform" is detected
    if 4 in class_ids:
        guard_present = True
        cap_detected = 0 in class_ids

        if guard_present and cap_detected and not previous_guard_present and not previous_cap_detected:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Send data to Firebase for proper uniformed guard with cap detected and timestamp
            data_to_push = {
                'Guard Present': True,
                'Uniform': 'Proper',
                'Timestamp': timestamp
            }
            db.collection('guards_monitoring').add(data_to_push)
        elif guard_present and not cap_detected and previous_guard_present and previous_cap_detected:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Send data to Firebase for proper uniformed guard without cap detected and timestamp
            data_to_push = {
                'Guard Present': True,
                'Uniform': 'Proper',
                'Timestamp': timestamp
            }
            db.collection('guards_monitoring').add(data_to_push)
        elif not previous_guard_present:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Send data to Firebase for guard present but no uniform detected and timestamp
            data_to_push = {
                'Guard Present': True,
                'Uniform': '',
                'Timestamp': timestamp
            }
            db.collection('guards_monitoring').add(data_to_push)

    elif previous_guard_present:
        # Guard not present, but was previously present
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Send data to Firebase for guard not present and timestamp
        data_to_push = {
            'Guard Present': False,
            'Uniform': '',
            'Timestamp': timestamp
        }
        db.collection('guards_monitoring').add(data_to_push)

    # Update previous states
    previous_guard_present = guard_present
    previous_cap_detected = cap_detected

    # #  # Plot results
    frame_ = results[0].plot()

    # # Visualize
    cv2.imshow('frame', frame_)

    # Check for the 'q' key to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
