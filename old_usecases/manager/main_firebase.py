import cv2
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/home/xloop/Bank_Alfalah/manager/smilefirebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the YOLOv8 model
model = YOLO('/home/xloop/Bank_Alfalah/manager/best.pt')

# Load video
video_path = '/home/xloop/Bank_Alfalah/manager/NVR_ch8_main_20230920150003_20230920160003.mp4'
cap = cv2.VideoCapture(video_path)

frame_rate = 25

class_present = False
class_start_time = None
class_end_time = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model.track(frame, persist=True)

    is_manager_detected = results[0].boxes.is_track

    if is_manager_detected:
        if not class_present:
            class_present = True
            class_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
    else:
        if class_present:
            class_present = False
            if class_start_time is not None:
                current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                class_end_time = current_datetime  # End time is the current timestamp
                class_start_time_dt = datetime.datetime.strptime(class_start_time, '%Y-%m-%d %H:%M:%S')
                class_end_time_dt = datetime.datetime.strptime(class_end_time, '%Y-%m-%d %H:%M:%S')
                duration = (class_end_time_dt - class_start_time_dt).total_seconds() / 60
                formatted_duration = "{:.2f}".format(duration)

                # Get the current date
                current_date = datetime.date.today()

                # Format the date as a string (YYYY-MM-DD)
                formatted_date = current_date.strftime('%Y-%m-%d')

                # Create a dictionary with the data to push to Firebase
                data_to_push = {
                    'Date': formatted_date,
                    'Start Time': class_start_time,
                    'End Time': class_end_time,
                    'Duration (m)': formatted_duration
                }

                # Push the data to Firebase
                db.collection('manager').add(data_to_push)

    frame_ = results[0].plot()

    cv2.imshow('frame', frame_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
