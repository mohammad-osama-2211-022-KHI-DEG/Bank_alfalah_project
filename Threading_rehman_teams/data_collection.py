import cv2
from ultralytics import YOLO
import os

# ------------------------------------FUNCTION TO BE USED FOR ANNOTATION----------------------------------------------
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
                boxes = result.boxes
                for box in boxes:
                    b = box.xyxyn  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    file.write(f'{model.names[int(c)]} {b[0][0]} {b[0][1]} {b[0][2]} {b[0][3]}\n')
# ------------------------------------FUNCTION TO BE USED FOR ANNOTATION----------------------------------------------


# Initialize YOLO model and video capture
model = YOLO('atm_cleanliness_best.pt')
video_path = 'videos/cleanliness.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

frame_count = 0

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
   
    # Detect objects in the frame
    results = model.predict(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            b = box.xyxyn  # Convert tensor to list
            c = box.cls
            print(f' {model.names[int(c)]} {b[0][0]} {b[0][1]} {b[0][2]} {b[0][3]}')
            
    # Process the frame results
    annotate_frame(frame_count, frame, results)

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
