import cv2
from ultralytics import YOLO
from collections import defaultdict
import csv
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Load the YOLOv8 model
model = YOLO('best.pt')

# Load video
video_path = 'NVR_ch8_main_20230920150003_20230920160003.mp4'
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

entry_times = {}  # Dictionary to store entry times
durations = {}   # Dictionary to store detected durations
tracked_list = []

# Create a CSV file to save object durations
csv_file = open("customer_waiting_duration.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Track ID", "Entry Time", "Exit Time", "Duration (seconds)"])

# Get frames per second of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Timer variables for alert
alert_timer = defaultdict(lambda: 0)
alert_interval = 2  # Alert every 2 seconds
alert_threshold = 5  # Alert if waiting time exceeds 5 seconds

# Variables for highest waited customer
max_wait_time = 0
max_wait_id = None

# Variables for total waiting time and total number of customers
total_waiting_time = 0
total_customers = 0

# Initialize Firebase Admin SDK
cred = credentials.Certificate("smilefirebase.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize frame for faster processing (adjust resolution as needed)
        frame = cv2.resize(frame, (1300, 800))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the current frame number and calculate the current time
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        for track_id, box in zip(track_ids, boxes):
            x, y, w, h = box

            # If the object is entering the frame
            if track_id not in entry_times:
                entry_times[track_id] = current_time
                total_customers += 1  # Increment total customer count

            # If the object is already in the frame
            else:
                exit_time = current_time
                duration = exit_time - entry_times[track_id]
                durations[track_id] = duration
                total_waiting_time += duration

                # Check if the waiting time exceeds the threshold
                if duration > alert_threshold:
                    # Check if it's time to show an alert
                    if current_time - alert_timer[track_id] > alert_interval:
                        print("\033[91m" + f"Alert: ID {track_id} has been waiting for {duration} seconds." + "\033[0m")
                        alert_timer[track_id] = current_time

                # Update the maximum waiting time and corresponding track ID
                if duration > max_wait_time:
                    max_wait_time = duration
                    max_wait_id = track_id

            cv2.putText(frame, f"ID: {track_id}, time: {durations.get(track_id, 0):.2f} sec", (int(x), int(y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Update entry times for tracked objects
        for track_id in tracked_list:
            if track_id not in track_ids:
                entry_times.pop(track_id, None)

        # Update tracked list
        tracked_list = track_ids

        # Display the annotated frame
        cv2.imshow("Customer Wait Time Tracking", frame)

        # Print the highest waited customer time and corresponding track ID
        if max_wait_id is not None:
            print(f"Highest waited customer (ID {max_wait_id}) waited for {max_wait_time:.2f} seconds.")

        # Print the total number of customers
        print(f"Total number of customers: {total_customers}")

        # Calculate and print the average waiting time
        if total_customers > 0:
            avg_waiting_time = total_waiting_time / total_customers
            average_waiting_time = avg_waiting_time / 60
            print(f"Average waiting time: {average_waiting_time:.2f} seconds")
        else:
            print("No customers detected.")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Write object durations to the CSV file
for track_id, duration in durations.items():
    entry_time = entry_times.get(track_id, 0)
    exit_time = entry_time + duration
    csv_writer.writerow([track_id, entry_time, exit_time, duration])

    # Add individual documents for each tracked object to Firestore
    doc_ref = db.collection('Customer_waiting_time').add({

        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Track_ID': track_id,
        # 'Entry_Time': entry_time,
        # 'Exit_Time': exit_time,
        'Duration': duration
    })

# Release the video capture object and close the display window
csv_file.close()
cap.release()
cv2.destroyAllWindows()
