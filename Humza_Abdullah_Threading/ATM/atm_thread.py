import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model_path = os.path.join('ATM/atm_best.pt')
model = YOLO(model_path)

def draw_detections_atm(frame, detections):
    trash_count = 0
    for result in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        threshold = 0.5
        if score > threshold and detections.names[int(class_id)].lower() == 'trash':
            trash_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, detections.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)
    
    return trash_count

def calculate_mess_level(trash_count):
    return round((trash_count / 1000) * 100, 2)

def process_frame(frame):
    results = model(frame)[0]
    prev_room_status = ""
    prev_mess_level = 0.0
    trash_count_yolo = draw_detections_atm(frame, results)
    mess_level = calculate_mess_level(trash_count_yolo)

    # Your clean/messy status logic here
    if trash_count_yolo == 0:
            room_status = "CLEAN"
    else:
            room_status = "MESSY"
     
    if room_status != prev_room_status or mess_level != prev_mess_level:
            print(f"Room Status Changed: {prev_room_status} -> {room_status}")
            print(f"Mess Level Changed: {prev_mess_level} -> {mess_level}")
            print(f"Trash Count:  {trash_count_yolo}")

    prev_room_status = room_status
    prev_mess_level = mess_level
    
 
    cv2.imshow("ATM Detections Thread", frame)
    
