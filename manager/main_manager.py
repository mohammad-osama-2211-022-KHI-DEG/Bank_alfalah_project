import cv2
import datetime
import time
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/home/xloop/Bank_Alfalah/Bank_alfalah_project/silent-space-358606-firebase-adminsdk-j9k7s-6bfcfa7775.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the YOLOv8 model
model = YOLO('/home/xloop/Bank_Alfalah/Bank_alfalah_project/manager/best.pt')

# Load video
video_path = '/home/xloop/Bank_Alfalah/Bank_alfalah_project/videos/manager_1.mp4'
cap = cv2.VideoCapture(video_path)

frame_rate = 25
duration_count = 0

block_triggered = False  # Initialize the flag to False

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
            class_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            duration_start_time = datetime.datetime.now() 

        current_time = datetime.datetime.now()
        elapsed_time = current_time - duration_start_time

        # Check if elapsed_time is greater than or equal to 5 seconds and the block hasn't been triggered yet
        if elapsed_time >= datetime.timedelta(seconds=5) and not block_triggered:
            # Trigger the block
            block_triggered = True

            logs = {
                'timeStamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'usecase': 'manager_floor_time',
                'country': 'pakistan',
                'branch': 'jinnah avenue',
                'city': 'islamabad',
                'duration': float("{:.2f}".format(duration_count)),
                'message': "manager's floor time limit (20 min) reached"
            }

            # Push the data to Firebase
            db.collection('logs').add(logs)
            

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
                duration_count = duration_count + duration
                print(duration_count)

                # Get the current date
                current_date = datetime.date.today()

                # Format the date as a string (YYYY-MM-DD)
                formatted_date = current_date.strftime('%Y-%m-%d')

                # Create a dictionary with the data to push to Firebase
                data_to_push = {
                   # 'view': 1,
                    #'id': 'PK-KHI-CLI-MANAGER',
                    'date': formatted_date,
                    'startTime': class_start_time,
                    'endTime': class_end_time,
                    'duration': float(formatted_duration),
                    'country': 'pakistan',
                    'branch': 'jinnah avenue',
                    'city': 'islamabad'
                }

                # Push the data to Firebase
                db.collection('manager_floor_time').add(data_to_push)

                # if c == 0:
                #     if duration_count > 0.5:
                #         logs = {
                #    # 'view': 1,
                #     #'id': 'PK-KHI-CLI-MANAGER',
                #     'date': formatted_date,
                #     'usecase': 'manager_floor_time',
                #     'country': 'pakistan',
                #     'branch': 'jinnah avenue',
                #     'city': 'islamabad',
                #     'duration': float("{:.2f}".format(duration_count)),
                #     'message': "manager's floor time limit (5 min) reached"
                # }

                # # Push the data to Firebase
                #         db.collection('logs').add(logs)
                #         c = 1
    
    print(elapsed_time)
    print(type(elapsed_time))


    frame_ = results[0].plot()

    cv2.imshow('Jinnah Avenue, Islamabad', frame_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()