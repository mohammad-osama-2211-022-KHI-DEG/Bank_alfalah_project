import cv2
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/home/xloop/Bank_Alfalah/Bank_alfalah_project/silent-space-358606-firebase-adminsdk-j9k7s-6bfcfa7775.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the YOLOv8 model
model = YOLO('/home/xloop/Bank_Alfalah/Bank_alfalah_project/best.pt')


#video_path_isl = '/home/xloop/Bank_Alfalah/Bank_alfalah_project/videos/manager_time_tracking.mp4'

# Paths to your video files
video_paths = ["/home/xloop/Bank_Alfalah/Bank_alfalah_project/videos/manager_view_2.mp4",
               "/home/xloop/Bank_Alfalah/Bank_alfalah_project/videos/manager_2.mp4"]
cap = cv2.VideoCapture(video_paths[0])
cap2 = cv2.VideoCapture(video_paths[1])
#cap3 = cv2.VideoCapture(video_path_isl)

frame_rate = 25

c = 0
duration_count = 0

block_triggered = False  # Initialize the flag to False

class_present = False
class_start_time = None
class_end_time = None

class_present2 = False
class_start_time2 = None
class_end_time2 = None

while True:
    success, frame = cap.read()
    success2, frame2 = cap2.read()
    #success3, frame3 = cap3.read()


    if not success and success2:
        break

    frame = cv2.resize(frame, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))
    #frame3 = cv2.resize(frame3, (640, 480))

    results = model.track(frame, persist=True)
    results2 = model.track(frame2, persist=True)
    #esults3 = model.track(frame3, persist=True)

    is_manager_detected = results[0].boxes.is_track
    is_manager_detected2 = results2[0].boxes.is_track
    #is_manager_detected3 = results3[0].boxes.is_track

    if is_manager_detected or is_manager_detected2:
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
                    'branch': 'clifton',
                    'city': 'karachi',
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
                    'branch': 'clifton',
                    'city': 'karachi'
                }

                # Push the data to Firebase
                db.collection('manager_floor_time').add(data_to_push)

                if c == 0:
                    if duration_count > 0.5:
                        logs = {
                   # 'view': 1,
                    #'id': 'PK-KHI-CLI-MANAGER',
                    'date': formatted_date,
                    'usecase': 'manager_floor_time',
                    'country': 'pakistan',
                    'branch': 'clifton',
                    'city': 'karachi',
                    'duration': float("{:.2f}".format(duration_count)),
                    'message': "manager's floor time limit (5 min) reached"
                }

                # Push the data to Firebase
                        db.collection('logs').add(logs)
                        c = 1

    # if is_manager_detected2:
    #     if not class_present2:
    #         class_present2 = True
    #         class_start_time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
    # else:
    #     if class_present:
    #         class_present2 = False
    #         if class_start_time2 is not None:
    #             current_datetime2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #             class_end_time2 = current_datetime  # End time is the current timestamp
    #             class_start_time_dt = datetime.datetime.strptime(class_start_time2, '%Y-%m-%d %H:%M:%S')
    #             class_end_time_dt = datetime.datetime.strptime(class_end_time2, '%Y-%m-%d %H:%M:%S')
    #             duration = (class_end_time_dt - class_start_time_dt).total_seconds() / 60
    #             formatted_duration = "{:.2f}".format(duration)

    #             # Get the current date
    #             current_date = datetime.date.today()

    #             # Format the date as a string (YYYY-MM-DD)
    #             formatted_date = current_date.strftime('%Y-%m-%d')

    #             # Create a dictionary with the data to push to Firebase
    #             data_to_push2 = {
    #                 'view': 2,
    #                 'id': 'PK-KHI-CLI-MANAGER',
    #                 'Date': formatted_date,
    #                 'Start Time': class_start_time,
    #                 'End Time': class_end_time,
    #                 'Duration (m)': formatted_duration
    #             }

    #             # Push the data to Firebase
    #             db.collection('manager').add(data_to_push2)
    # if is_manager_detected3:
    #     if not class_present2:
    #         class_presen2t = True
    #         class_start_time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
    # else:
    #     if class_present2:
    #         class_present2 = False
    #         if class_start_time2 is not None:
    #             current_datetime2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #             class_end_time = current_datetime  # End time is the current timestamp
    #             class_start_time_dt2 = datetime.datetime.strptime(class_start_time, '%Y-%m-%d %H:%M:%S')
    #             class_end_time_dt2 = datetime.datetime.strptime(class_end_time, '%Y-%m-%d %H:%M:%S')
    #             duration2 = (class_end_time_dt2 - class_start_time_dt2).total_seconds() / 60
    #             formatted_duration2 = "{:.2f}".format(duration2)

    #             # Get the current date
    #             current_date2 = datetime.date.today()

    #             # Format the date as a string (YYYY-MM-DD)
    #             formatted_date2 = current_date2.strftime('%Y-%m-%d')

    #             # Create a dictionary with the data to push to Firebase
    #             data_to_push2 = {
    #                 'id': 'PK-ISL-JA-MANAGER',
    #                 'Date': formatted_date2,
    #                 'Start Time': class_start_time2,
    #                 'End Time': class_end_time2,
    #                 'Duration (m)': formatted_duration2
    #             }

    #             # Push the data to Firebase
    #             db.collection('manager').add(data_to_push2)

    annotated_frame = results[0].plot()
    annotated_frame2 = results2[0].plot()
    #annotated_frame3 = results3[0].plot()

    cv2.imshow("karachi, clifton view-1", annotated_frame)
    cv2.imshow("karachi, clifton view-2", annotated_frame2)
    #cv2.imshow("islamabad, jinnah avenue", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()