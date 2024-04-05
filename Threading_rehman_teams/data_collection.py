import cv2
from ultralytics import YOLO
import os

# Function to process a single frame's results and save them
def annotate_frame(frame_count, frame, results):
    if len(results[0].boxes.xyxyn) > 0:
        # Create directories if they don't exist
        os.makedirs('./images/', exist_ok=True)
        os.makedirs('./labels/', exist_ok=True)

        image_filename = f'./images/frame_{frame_count}.jpg'
        label_filename = f'./labels/frame_{frame_count}.txt'

        # Save image
        cv2.imwrite(image_filename, frame)

        # Save labels
        with open(label_filename, 'w') as file:
            for result in results:
                for i in range(len(result.boxes.xyxy)):
                    file.write(f'{result.names[i]} {result.boxes.xyxyn[i][0]} {result.boxes.xyxyn[i][1]} {result.boxes.xyxyn[i][2]} {result.boxes.xyxyn[i][3]}\n')



# Initialize YOLO model and video capture
model = YOLO('new_best.pt')
video_path = 'ATM_working.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

frame_count = 0

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    results = model.predict(frame)

    # Process the frame results
    annotate_frame(frame_count, frame, results)

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
