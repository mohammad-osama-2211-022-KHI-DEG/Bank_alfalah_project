import subprocess
import cv2
import re
from datetime import datetime
import time
import firebase_admin
from firebase_admin import credentials, firestore

class_present = False
class_start_time = None
class_end_time = None
formatted_duration = 0.0

cred = credentials.Certificate('/home/xloop/Bank_Alfalah/Bank_alfalah_project/silent-space-358606-firebase-adminsdk-j9k7s-6bfcfa7775.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

yolo_command = "yolo task=detect mode=predict model=yolov8m.pt show=True conf=0.5 source=/home/xloop/Bank_Alfalah/Bank_alfalah_project/videos/office.mp4"
cap = cv2.VideoCapture("/home/xloop/Bank_Alfalah/Bank_alfalah_project/videos/office.mp4")

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit(1)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

cup_threshold = 1
person_threshold = 2
laptop_threshold = 1
chair_threshold = 1
tv_threshold = 1

previous_cleanliness_status = None
cleanliness_changed = False

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))
frame_count = 0

# def save_cleanliness_result(clean_scene):
#     db = firestore.client()
#     collection_name = "cleanliness"
#     formatted_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     result_data = {
#         'state': "Room is clean" if clean_scene == 'Clean' else "Room is Messy",
#         'ArrivalTime': formatted_datetime,
#         'country': 'pakistan', 
#         'branch': 'clifton',
#         'city': 'karachi'
#     }

    # try:
    #     new_document_ref, new_document_id = db.collection(collection_name).add(result_data)
    #     print(f"Data stored in a new document with ID: {new_document_id}")
    # except Exception as e:
    #     print(f"Error storing data in Firestore: {e}")

try:
    yolo_process = subprocess.Popen(
        yolo_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    for line in yolo_process.stdout:
        line = line.strip()
        print(line)

        object_counts = {
            "cup": 0,
            "person": 0,
            "laptop": 0,
            "chair": 0,
            "tv": 0,
        }

        counts = re.findall(r"(\d+) (\w+)", line)
        for count, obj in counts:
            if obj in object_counts:
                object_counts[obj] = int(count)

        ret, frame = cap.read()
        if ret:
            if (
                object_counts["cup"] >= cup_threshold
                or object_counts["person"] >= person_threshold
                or object_counts["laptop"] >= laptop_threshold
                and object_counts["chair"] >= chair_threshold
                and object_counts["tv"] >= tv_threshold
            ):
                cleanliness_status = "Messy"
            else:
                cleanliness_status = "Clean"

            if cleanliness_status != previous_cleanliness_status:
                class_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if cleanliness_changed:
                    formatted_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    result_data = {
                        'state': previous_cleanliness_status,
                        'ArrivalTime': formatted_datetime,
                        'country': 'pakistan', 
                        'branch': 'jinnah avenue',
                        'city': 'islamabad'
                    }
                    db.collection('cleanliness').add(result_data)

                previous_cleanliness_status = cleanliness_status
                cleanliness_changed = True

                if cleanliness_status == "Messy":
                    logs = {
                        #'duration': duration,
                        'status': 'Messy',
                        'timeStamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'country': 'pakistan', 
                        'branch': 'jinnah avenue',
                        'city': 'islamabad',
                        'usecase': 'cleanliness',
                        'message': 'messy branch detected'
                    }
                    db.collection('logs').add(logs)
                        

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            overlay_text = f"Frame {frame_count}: Room: {cleanliness_status}  Time: {current_time}"
            cv2.putText(frame, overlay_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_video.write(frame)
            cv2.imshow("Cleanliness Status", frame)
            print(f"Frame {frame_count}: Time: {current_time}, Room: {cleanliness_status}")
            frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    yolo_process.terminate()
    yolo_process.wait()

except subprocess.CalledProcessError as e:
    print(f"Error running YOLO command: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

cap.release()
output_video.release()
cv2.destroyAllWindows()

# if cleanliness_changed:
#     save_cleanliness_result(previous_cleanliness_status)
