
import os
from ultralytics import YOLO
import cv2

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

video_path = os.path.join('9.mp4')
#video_path =video_url
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('atm_best.pt')
model = YOLO(model_path)
threshold = 0.5

cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detections", 800, 600)

prev_room_status = ""
prev_mess_level = 0.0

while ret:
    results = model(frame)[0]

    trash_count_yolo = draw_detections_atm(frame, results)

    mess_level = calculate_mess_level(trash_count_yolo)

    if trash_count_yolo == 0:
        room_status = "CLEAN"
    else:
        room_status = "MESSY"

    if room_status != prev_room_status or mess_level != prev_mess_level:
        print(f"Room Status Changed: {prev_room_status} -> {room_status}")
        print(f"Mess Level Changed: {prev_mess_level} -> {mess_level}")

    prev_room_status = room_status
    prev_mess_level = mess_level

    # Add overlay text with Trash Count
    status_text = f"Room Status: {room_status}"
    count_text = f"Trash Count: {trash_count_yolo}"
    level_text = f"Mess Level: {mess_level}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, level_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    print("Trash Count: ", trash_count_yolo ," Mess Level: " ,mess_level)

    out.write(frame)
    cv2.imshow("YOLO Detections", frame)

    # Introduce a delay to make the video display smoother
    cv2.waitKey(1)

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
