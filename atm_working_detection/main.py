import cv2
from ultralytics import YOLO
from datetime import datetime
import time
from datetime import datetime, timezone, timedelta
import httpx
import requests

headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJhbmlxYS5tYXNvb2RAeGxvb3BkaWdpdGFsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwNjAwMzgyNiwiZXhwIjoxNzA2MzAzODI2fQ.sXSY31PZ7ShamadKxcfTTerJgsF9pYrHdbQqABpJTy8",
        "X-XSRF-TOKEN": "fccfbe65-6422-4cf6-a302-fb6ebc90ce8e",
        "X-SERVER-TO-SERVER" : "true"
    }

url = 'http://13.235.71.140:5000/atm-functionality'
session = httpx.Client()

# ATM functionality check 
def detect_atm_usage(model_path, video_path, target_fps):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / target_fps)
    counter = 0
    count = 0
    atm_detected = False
    workingStatus = True
    total_working_count=0
    total_notworking_count=0
    complaintBoxAvailable = False
    telephoneAvailable = False

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print(f"Working with frame {frame_no}, no frame detected")
            break
        else:
            results = model.track(source=frame, show=True, project='./result', tracker="bytetrack.yaml", conf=0.4)
            a = None
            for result in results:
                class_name = result.boxes.cls.tolist()
                conf_name = result.boxes.conf.tolist()
                res = dict(zip(class_name, conf_name))
                # check for complain-box and telephone
                alert_complainbox_telephone= complainbox_telephone(class_name)
                complaintBoxAvailable = alert_complainbox_telephone["complaintBoxAvailable"]
                telephoneAvailable = alert_complainbox_telephone["telephoneAvailable"]
                if len(res) != 0 and atm_detected == False:
                    if atm_detected == False and 0.0 in class_name and res[0.0] >= 0.4:
                        # print("ATM Card is Detected")
                        atm_detected = True
                        count = 0
                    else:
                        pass
                elif len(res) != 0 and atm_detected == True:
                    if 1.0 in class_name and res[1.0] >= 0.8:
                        # print("Cash is Detected")
                        workingStatus= True
                        total_working_count +=1
                        atm_detected = False
                        timestamp = Time()
                        data = data_preparation(workingStatus,total_working_count,total_notworking_count,complaintBoxAvailable,telephoneAvailable,timestamp) 
                        response = requests.post(url, json=data, headers=headers)
                        print("Endpoint response status code:", response.status_code)
                    elif count >= 1000:
                        print("ATM detection timeout reached. Exiting...")
                        counter += 1
                        total_notworking_count +=1
                        # Counting for 6 cash not detected
                        if counter >= 1:
                            workingStatus= False
                            counter = 0
                            timestamp = Time()
                            data = data_preparation(workingStatus,total_working_count,total_notworking_count,complaintBoxAvailable,telephoneAvailable,timestamp) 
                            response = requests.post(url, json=data, headers=headers)
                            print("Endpoint response status code:", response.status_code)
                        atm_detected = False
                    else:
                        count +=1
                elif atm_detected == True:
                    if count >= 1000:
                        print("ATM detection timeout reached. Exiting...")
                        counter += 1
                        total_notworking_count +=1
                        # Counting for 6 cash not detected
                        if counter >= 1:
                            workingStatus= False
                            counter = 0
                            timestamp = Time()
                            data = data_preparation(workingStatus,total_working_count,total_notworking_count,complaintBoxAvailable,telephoneAvailable,timestamp)
                            response = requests.post(url, json=data, headers=headers)
                            print("Endpoint response status code:", response.status_code)
                    
                        atm_detected = False
                    else:
                        count +=1
                else:
                    pass
            # Increment frame counter
        frame_no += 1
    # Release video capture
    cap.release()

# Check for complain-box and telephone
def complainbox_telephone(class_name):
    # Complain-box & Telephone are present
    if 2.0 in class_name and 3.0 in class_name:
        alert= {"complaintBoxAvailable": True,
        "telephoneAvailable": True}
        return alert
    # Complain-box is present & Telephone is not present
    elif 2.0 in class_name and 3.0 not in class_name:
        alert= {"complaintBoxAvailable": True,
        "telephoneAvailable": False}
        return alert
    # Complain-box is not present & Telephone is present
    elif 2.0 not in class_name and 3.0 in class_name:
        alert= {"complaintBoxAvailable": False,
        "telephoneAvailable": True}
        return alert 
    # Complain-box & Telephone are not present
    elif 2.0 not in class_name and 3.0 not in class_name:
        alert= {"complaintBoxAvailable": False,
        "telephoneAvailable": False}
        return alert   
def Time():
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S%z')
    return formatted_date
def data_preparation(workingStatus,total_working_count,total_notworking_count,complaintBoxAvailable,telephoneAvailable,timestamp):
    data = {
    "workingStatus": workingStatus, 
    "totalSuccessfulTransaction" :total_working_count,
    "totalUnsuccessfulTransaction": total_notworking_count ,
    "complaintBoxAvailable": complaintBoxAvailable, 
    "telephoneAvailable": telephoneAvailable, 
    "timestamp": timestamp,
    "country":"pakistan",
    "city": "karachi",
    "branch": "clifton"
    } 
    return data

if __name__ == '__main__':
    model_path = 'atm_functionality_best.pt'
    video_path = "YouCut_20240114_230531505.mp4"
    target_fps = 8
    detect_atm_usage(model_path, video_path, target_fps)
