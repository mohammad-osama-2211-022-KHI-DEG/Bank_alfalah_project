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


# ------------------------------------FUNCTION TO BE USED FOR BB images----------------------------------------------
def save_images_with_boxes(frame_count, frame, results):
    if len(results[0].boxes.xyxyn) > 0:
        bb_images_dir = './bb_images/'
        os.makedirs(bb_images_dir, exist_ok=True)

        for i, result in enumerate(results):
            image = frame.copy()
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                x1, y1, x2, y2 = map(int, [b[0][0], b[0][1], b[0][2], b[0][3]])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(f'{bb_images_dir}/frame_{frame_count}_obj_{model.names[int(c)]}.jpg', image)
# ------------------------------------FUNCTION TO BE USED FOR BB images----------------------------------------------

'''
Code below will be used for demonstration purposes.
functions above are ready to be integrated in the refactored code
'''


# Initialize YOLO model and video capture
model = YOLO('suspecious_updated.pt')
video_path = 'videos/suspecious.mp4'  # Replace with your video path
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
    
    save_images_with_boxes(frame_count, frame, results)


    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
