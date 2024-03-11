# # import os

# # from ultralytics import YOLO
# # import cv2


# # VIDEOS_DIR = os.path.join('.', 'video')

# # video_path = os.path.join(VIDEOS_DIR, 'merged_new.mp4')
# # video_path_out = '{}_out.mp4'.format(video_path)

# # cap = cv2.VideoCapture(video_path)
# # ret, frame = cap.read()
# # H, W, _ = frame.shape
# # out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# # model_path = os.path.join('new_best.pt')

# # # Load a model
# # model = YOLO(model_path)  # load a custom model

# # threshold = 0.5

# # while ret:

# #     results = model(frame)[0]

# #     for result in results.boxes.data.tolist():
# #         x1, y1, x2, y2, score, class_id = result

# #         if score > threshold:
# #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
# #             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# #     out.write(frame)
# #     ret, frame = cap.read()

# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()



# import os
# from ultralytics import YOLO
# import cv2

# #VIDEOS_DIR = os.path.join('.', 'video')

# video_path = os.path.join('merged_new.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# model_path = os.path.join('new_best.pt')

# # Load a model
# model = YOLO(model_path)  # load a custom model

# threshold = 0.5

# while ret:

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     out.write(frame)
#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     ret, frame = cap.read()

# cap.release()
# out.release()
# cv2.destroyAllWindows()


import cv2
import subprocess
import re
import requests
from datetime import datetime, timezone, timedelta
import threading

#url = "rtsp://192.168.100.7:1578/video"


def build_payload(formatted_date, mess_level, room_status):
    return {
        "country": "pakistan",
        "branch": "clifton",
        "city": "karachi",
        "timestamp": formatted_date,
        "cleanlinessState": room_status,
        "levelOfMess": mess_level,
    }

def post_data_to_endpoint(endpoint_url, tag, headers, payload):
    response = requests.post(endpoint_url.format(TAG=tag), json=payload, headers=headers)
    if response.status_code == 200:
        print("Data posted successfully!")
    else:
        print(f"Failed to post data. Status code: {response.status_code}")


def initialize_video_writer(output_path, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

def initialize_yolo_process(weights_path, video_source):
    yolo_command = f"yolo task=detect mode=predict model={weights_path} show=True conf=0.5 source={video_source}"
    return subprocess.Popen(
        yolo_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

def capture_and_process_yolo_output(yolo_output):
    match = re.search(r'(\d+)\s*Trash', yolo_output)
    if match:
        trash_count = int(match.group(1))
        return trash_count
    return 0

def calculate_mess_level(trash_count):
    return (trash_count / 1000) * 100



def read_and_display_frames(cap, out, yolo_process):
    global prev_room_status, prev_mess_level
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on the frame
        yolo_output = yolo_process.stdout.readline().strip()

        # Process YOLO output
        trash_count = capture_and_process_yolo_output(yolo_output)

        current_datetime = datetime.now(timezone(timedelta(hours=5)))
        formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
        mess_level = calculate_mess_level(trash_count)

        if trash_count == 0:
            room_status = "CLEAN"
        else:
            room_status = "MESSY"

        if room_status != prev_room_status or mess_level != prev_mess_level:
            print(f"Room Status Changed: {prev_room_status} -> {room_status}")
            print(f"Room Status Changed: {prev_mess_level} -> {mess_level}")
            print("Trash Count: ", trash_count)
            # payload = build_payload(formatted_date, mess_level, room_status)
            # print(payload)

        prev_room_status = room_status
        prev_mess_level = mess_level

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2
        overlay_text1 = f"Room Status: {room_status}"
        overlay_text2 = f"Trash Count: {trash_count}"
        overlay_text3 = f"Mess Level: {mess_level}%"

        cv2.putText(frame, overlay_text1, (10, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(frame, overlay_text2, (10, 60), font, font_scale, font_color, font_thickness)
        cv2.putText(frame, overlay_text3, (10, 90), font, font_scale, font_color, font_thickness)

        out.write(frame)

    cap.release()
    out.release()
    yolo_process.terminate()
    cv2.destroyAllWindows()


def main():
    global prev_room_status, prev_mess_level
    video_path = 'Atm_Final.mp4'
    output_path = 'output_video.mp4'
    yolo_weights_path = 'new_best.pt'
    endpoint_url = "http://13.233.56.158:5000/cleanliness/{TAG}"
    tag = "atm"
    jwt_token = "11eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJhbmlxYW1hc29vZDExMUBnbWFpbC5jb20iLCJhdXRob3JpdGllcyI6IkNSRUFURV9VU0VSLFZJRVciLCJpYXQiOjE3MDc5MDQ0NDAsImV4cCI6MTczNDE3MDA0MH0.QH8ZgFAWPdEZHfYt3uvw3JShHVijiyniTcdOMxfH9Y8"
    notification_url = "http://13.233.56.158:5000/notification"
    headers = {
        "Content-Type": "application/json",
        "Authorization": jwt_token,
        "X-SERVER-TO-SERVER": "true"
    }

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = initialize_video_writer(output_path, frame_width, frame_height)

    yolo_process = initialize_yolo_process(yolo_weights_path, video_path)

    prev_room_status = None
    prev_mess_level = None

    # Start a thread for reading and displaying frames
    frame_thread = threading.Thread(target=read_and_display_frames, args=(cap, out, yolo_process))
    frame_thread.start()

    # Wait for the frame thread to finish
    frame_thread.join()

    # Continue with the rest of the main function if needed


if __name__ == "__main__":
    main()


# def main():
#     video_path = url
#     output_path = 'output_video.mp4'
#     yolo_weights_path = 'new_best.pt'
#     endpoint_url = "http://13.233.56.158:5000/cleanliness/{TAG}" 
#     tag = "atm"
#     jwt_token = "11eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJhbmlxYW1hc29vZDExMUBnbWFpbC5jb20iLCJhdXRob3JpdGllcyI6IkNSRUFURV9VU0VSLFZJRVciLCJpYXQiOjE3MDc5MDQ0NDAsImV4cCI6MTczNDE3MDA0MH0.QH8ZgFAWPdEZHfYt3uvw3JShHVijiyniTcdOMxfH9Y8"
#     notification_url = "http://13.233.56.158:5000/notification"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": jwt_token,
#         "X-SERVER-TO-SERVER": "true"
#     }

#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     out = initialize_video_writer(output_path, frame_width, frame_height)

#     yolo_process = initialize_yolo_process(yolo_weights_path, video_path)
#     prev_room_status = None 
#     prev_mess_level = None
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Run YOLO on the frame
#         yolo_output = yolo_process.stdout.readline().strip()

#         # Process YOLO output
#         trash_count = capture_and_process_yolo_output(yolo_output)

        
#         current_datetime = datetime.now(timezone(timedelta(hours=5)))
#         formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
#         # Calculate mess level
#         mess_level = calculate_mess_level(trash_count)

#         # Use mess level for your comparison (room is clean if trash count is 0)
#         if trash_count == 0:
#             room_status = "CLEAN"
#         else:
#             room_status = "MESSY"

#         if room_status != prev_room_status or mess_level != prev_mess_level:
#             print(f"Room Status Changed: {prev_room_status} -> {room_status}")
#             print(f"Room Status Changed: {prev_mess_level} -> {mess_level}")
#             # You can perform actions here when the room status changes
#             payload = build_payload(formatted_date, mess_level, room_status)
#             # post_data_to_endpoint(endpoint_url, tag, headers, payload)
#             print(payload)
            
            

#         prev_room_status = room_status  # Update previous room status
#         prev_mess_level = mess_level 
#         if mess_level > 0.5:
                    
#                     notification_message = f"High level of mess detected! {mess_level}% mess."
#                     notification_payload = {
#                         "timestamp": formatted_date,
#                         "message": notification_message,
#                         "country": "pakistan",
#                         "city": "karachi",
#                         "branch": "clifton",
#                         "usecase": "atm_cleanliness",
#                     }
#                     # response = requests.post(notification_url, json=notification_payload, headers=headers)
#                     # print("Notification",response)
#                     # print(response.content)
#                     # if response.status_code == 200:
#                     #     print("Notification sent successfully!")
#                     # else:
#                     #     print(f"Failed to send notification. Status code: {response.status_code}")
        
#         print(f"Trash count: {trash_count}, Mess Level: {mess_level}, Room Status: {room_status}, Previous Status: {prev_room_status}, Previous Mess Level {prev_mess_level}")
        
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_color = (255, 255, 255)  # White color in BGR format
#         font_thickness = 2
#         overlay_text1 = f"Room Status: {room_status}"
#         overlay_text2 = f"Trash Count: {trash_count}"
#         overlay_text3 = f"Mess Level: {mess_level}%"

#         cv2.putText(frame, overlay_text1, (10, 30), font, font_scale, font_color, font_thickness)
#         cv2.putText(frame, overlay_text2, (10, 60), font, font_scale, font_color, font_thickness)
#         cv2.putText(frame, overlay_text3, (10, 90), font, font_scale, font_color, font_thickness)
    
        
#         # Write the frame to the output video
#         out.write(frame)

#     cap.release()
#     out.release()
#     yolo_process.terminate()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
