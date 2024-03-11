import os
from ultralytics import YOLO
import cv2
import requests
from datetime import datetime, timezone, timedelta
import time


username = 'usama.xloop'
password = 'Xloop12345'
ip_address = '192.168.6.13'
video_url = f"rtsp://{username}:{password}@{ip_address}:554/cam/realmonitor?channel=1&subtype=1"


def draw_detections(frame, detections):
    for result in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        threshold = 0.5
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, detections.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)



status=[]
def check_status_and_avglevel(detected_tables_info):
    clean_status = "CLEAN"
    if detected_tables_info:
        for table_info in detected_tables_info.values():
            if table_info['status'] == "MESSY":
                clean_status = "MESSY"
                break  # Stop iterating if any table is messy
        status.append(clean_status)
        
        last_added_item = list(detected_tables_info.values())[-1]
        # Ensure last_added_item is not None before accessing its keys
        if last_added_item is not None:
            avglevelMess_value = last_added_item.get('avglevelMess', None)
            if avglevelMess_value is not None:
                if len(status) > 2:
                    status.pop(0)
                if len(set(status)) == 2:
                    current_datetime = datetime.now(timezone(timedelta(hours=5)))
                    formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00') 
                    cleanliness_status = status[-1]
                    payload = build_payload(formatted_date, cleanliness_status, avglevelMess_value)
                    message = f"Room Status is now: {clean_status}"
                    notification = build_notification(formatted_date, message, "pakistan", "karachi", "clifton",
                                                       "branch_cleanliness")
                    post_data_to_endpoint(endpoint_url, tag, headers, payload)
                    post_notification(notification_url, headers, notification)
                    print("Data should be posted:", status, " Overall Average:", avglevelMess_value)
                    print(payload)
                    print(notification)

        print("Current Stats:\nLevel of Mess:", avglevelMess_value, " Room Status:", clean_status, "LIST:", status)


def build_payload(formatted_date, clean_status,avglevelofmess):
    return {
                    "timestamp": formatted_date,
                    "cleanlinessState": clean_status,
                    "country": "pakistan",
                    "city": "karachi",
                    "branch": "clifton",        
                    "levelOfMess": avglevelofmess
    }
current_datetime = datetime.now(timezone(timedelta(hours=5)))
formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')    
def build_notification(formatted_date,message,country,city,branch,usecase):
    return {
                        "timestamp": formatted_date,
                        "message": message,
                        "country": country,
                        "city": city,
                        "branch": branch,
                        "usecase": usecase
                    }
endpoint_url = "http://13.126.160.174:5000/cleanliness/{TAG}"
notification_url = "http://13.126.160.174:5000/notification"
tag = "branch"
jwt_token = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU_1"
headers = {
        "Content-Type": "application/json",
        "Authorization": jwt_token,
        "X-SERVER-TO-SERVER": "true"
    }
def post_data_to_endpoint(endpoint_url, tag, headers, payload):
    response = requests.post(endpoint_url.format(TAG=tag), json=payload, headers=headers) 
    print(f"Response: {response.status_code}")
    if response.status_code == 200:
        print("Data posted successfully!")
    else:
        print(f"Failed to post data. Status code: {response.status_code}")
        
def post_notification(notification_url,headers, notification):
    response = requests.post(notification_url, json=notification,headers=headers)
    if response.status_code == 200:
        print("Notification posted successfully!")
    else:
        print(f"Failed to post notification. Status code: {response.status_code}")
        
def process_frame(frame, model_objects, model_tables, output_width, output_height, threshold,
                  detected_tables_info, previous_status, out, tables_and_windows):
    H, W, _ = frame.shape
    
    resized_frame = cv2.resize(frame, (output_width, output_height))
    resized_frame_objects = resized_frame  # Initialize here as an empty frame
    
    total_level_of_mess = 0.0
    num_detected_tables = 0

    results_tables = model_tables(resized_frame)[0]

    for window_name in tables_and_windows:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow(window_name)


    tables_and_windows = []
    
    # for window_name in reversed(tables_and_windows):
    #     if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
    #         cv2.destroyWindow(window_name)

    # tables_and_windows = []


    curr_table_count = len(results_tables.boxes)
    
    # Initialize prev_table_count before using it
    prev_table_count = getattr(process_frame, 'prev_table_count', 0)

    if curr_table_count < prev_table_count:
        for i in range(curr_table_count, prev_table_count):
            window_name = f"Table ID {i + 1}"
            cv2.destroyWindow(window_name)

    setattr(process_frame, 'prev_table_count', curr_table_count)

    detected_tables_info = {}
    detected_objects_info = {}

    for i, result in enumerate(results_tables.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(resized_frame, results_tables.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Convert the coordinates to match the original frame size
            x1_orig = int(x1 * (W / output_width))
            y1_orig = int(y1 * (H / output_height))
            x2_orig = int(x2 * (W / output_width))
            y2_orig = int(y2 * (H / output_height))

            # Crop the region of the table from the original frame
            table_region = frame[y1_orig:y2_orig, x1_orig:x2_orig]
            # Timestamp
            current_datetime = datetime.now(timezone(timedelta(hours=5)))
            formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
            
            # Save information about detected tables
            detected_tables_info[i] = {'id': i + 1, 'coordinates': (x1_orig, y1_orig, x2_orig, y2_orig), 'status': "CLEAN",'timestamp': formatted_date}
            # Apply object detection on the original frame
            results_objects = model_objects(frame)[0]
            for result in results_objects.boxes.data.tolist():
                x1_obj, y1_obj, x2_obj, y2_obj, score_obj, class_id_obj = result

                # Save information about detected objects
                detected_objects_info[(x1_obj, y1_obj, x2_obj, y2_obj)] = {'coordinates': (x1_obj, y1_obj, x2_obj, y2_obj),
                                                                            'class_id': class_id_obj}
            
            # Draw detections for objects on the original frame
            draw_detections(frame, results_objects)

            # Resize the frame with object detections to the desired output dimensions
            resized_frame_objects = cv2.resize(frame, (output_width, output_height))

            # Show the cropped table in the replicated window
            cv2.imshow(f"Table ID {i + 1}", table_region)

    results_objects = model_objects(frame)[0]
    
              
    for table_id, table_info in detected_tables_info.items():
        obj_count = 0
        for obj_coords, obj_info in detected_objects_info.items():
                    # Check for the intersection of bounding boxes
            if not (obj_coords[2] < table_info['coordinates'][0] or
                    obj_coords[0] > table_info['coordinates'][2] or
                    obj_coords[3] < table_info['coordinates'][1] or
                    obj_coords[1] > table_info['coordinates'][3]):
                     obj_count += 1
        # Level of Mess calculation
        level_of_mess = (obj_count / 1000) * 100
        
        # Avg Level of Mess
        total_level_of_mess += level_of_mess
        num_detected_tables += 1
        avg_level_of_mess = total_level_of_mess / num_detected_tables
        
        table_info['avglevelMess'] = round(avg_level_of_mess,2)
        # Assign object count to the table
        table_info['obj_count'] = obj_count
        table_info['levelofMess'] = round(level_of_mess,2)
    
        

        # Check the size of the table and set the status accordingly
        if (table_info['coordinates'][2] - table_info['coordinates'][0] > 400 and
                table_info['coordinates'][3] - table_info['coordinates'][1] > 200):
            if obj_count > 14:
                table_info['status'] = "MESSY"  # Set status to messy
        else:
            if obj_count > 7:
                table_info['status'] = "MESSY"  # Set status to messy
                
        print(table_info)
        

            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        # Overlay text on the video
        cv2.putText(resized_frame_objects, f"Table ID {table_info['id']} - Status: {table_info['status']}",
                    (10, 30 * (table_id + 1)), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    
    check_status_and_avglevel(detected_tables_info)
       
    cv2.imshow("YOLO Detections", resized_frame_objects)
    out.write(resized_frame_objects)

    return detected_tables_info, previous_status, tables_and_windows


def main():
    #video_path = os.path.join('video', 'Small.mp4')
    #url = "rtsp://192.168.100.7:1578/video"
    video_path= video_url
    video_path_out = '{}_out.mp4'.format(video_path)

    
   

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    H, W, _ = frame.shape
    output_width = 640
    output_height = 480
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (output_width, output_height))

    #new Model
    model_path_objects = os.path.join('objects_weights', 'best.pt')
    model_path_tables = os.path.join('table_weights', 'best.pt')

    model_objects = YOLO(model_path_objects)
    model_tables = YOLO(model_path_tables)
    threshold = 0.5

    cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detections", 800, 600)

    prev_table_count = len(model_tables(frame)[0].boxes)
    detected_tables_info = {}
    detected_objects_info = {}

    previous_status = "CLEAN"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    level_of_mess = 0.0
    total_level_of_mess = 0.0
    num_detected_tables = 0
    tables_and_windows = []  # Initialize the list here
    
    frame_count = 0
    display_frame_interval = 30 
    while ret:
        detected_tables_info,previous_status, tables_and_windows = process_frame(
            frame, model_objects, model_tables, output_width, output_height, threshold,
            detected_tables_info, previous_status, out, tables_and_windows
        )
         # Reset the timer after processing a frame
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        ret, frame = cap.read()
        
        if frame_count % display_frame_interval == 0:
            cv2.imshow("YOLO Detections", frame)
            frame_count += 1
        else:
            frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
