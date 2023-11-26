import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict
import csv
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime


# Load the YOLOv8 model
model = YOLO('/home/xloop/Bank_Alfalah/Bank_alfalah_project/customer_waiting_time/best.pt')

# Load video
video_path = '/home/xloop/Bank_Alfalah/Bank_alfalah_project/videos/manager_2.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("/home/xloop/Bank_Alfalah/Bank_alfalah_project/silent-space-358606-firebase-adminsdk-j9k7s-6bfcfa7775.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture(video_path)



entry_times = {}  # Dictionary to store entry times
durations = []  # List to store detected durations
tracked_list = []
data_to_push = {}
logs_dict = {}
logs_track_list = []

# Create a CSV file to save object durations
csv_file = open("customer_waiting_duration.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Track ID", "Entry Time", "Exit Time", "Duration (seconds)"])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # Resize frame for faster processing (adjust resolution as needed)
        frame = cv2.resize(frame, (640, 480))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the current time
        current_time = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))


        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        print(track_ids)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Update entry times and calculate durations
        for track_id in track_ids:
            if track_id not in entry_times:
                entry_times[track_id] = current_time
                data_to_push = {
                        'id': 'PK-ISL-JA-'+ str(track_id),
                        #'track_id': track_id,
                        'time': formatted_time,
                        'status': 'present',
                        'country': 'pakistan',
                        'city': 'islamabad',
                        'branch': 'jinnah avenue'

                    }
                db.collection('customer_wait_time').add(data_to_push)

                for track_id in list(entry_times.keys()):
                    if track_id not in tracked_list:
                        if track_id not in track_ids:
                            exit_time = current_time
                            duration = exit_time - entry_times[track_id]
                            #durations.append(track_id, entry_times[track_id], exit_time, duration)
                            data_to_push = {
                                'id': 'PK-ISL-JA-'+ str(track_id),
                                #'track_id': track_id,
                                'time': formatted_time,
                                'status': 'exit',
                                'country': 'pakistan',
                                'city': 'islamabad',
                                'branch': 'jinnah avenue'

                            }
                            db.collection('customer_wait_time').add(data_to_push)
                            tracked_list.append(track_id)


        print(entry_times)

        # Find and remove objects that have exited the frame
        for track_id in entry_times.keys():
            duration = current_time - entry_times[track_id]
            if duration >= 30 and track_id not in tracked_list:
                formatted_duration = round(duration, 2)
                logs_dict = {
                        'id': track_id,
                        'usecase': 'customer_wait_time',
                        #'track_id': track_id,
                        'timeStamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'duration': duration,
                        'country': 'pakistan',
                        'city': 'islamabad',
                        'branch': 'jinnah avenue',
                        'message': 'PK-ISL-JA-'+ str(track_id) + ' is waiting for ' + str(round(formatted_duration/ 60, 2)) + 'm in waiting area'
                    }
                
                db.collection('logs').add(logs_dict)
                logs_track_list.append(track_id)

        
                

        # Plot the tracks
        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)

        #     # Draw the tracking lines
        #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("Customer Wait Time Tracking, Jinnah Avenue, Islamabad", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# d = []
# d.append(durations[0])

# # Write object durations to the CSV file
# for track_id, entry_time, exit_time, duration in durations:
#     csv_writer.writerow([track_id, entry_time, exit_time, duration])

# Release the video capture object and close the display window
csv_file.close()
cap.release()
cv2.destroyAllWindows()