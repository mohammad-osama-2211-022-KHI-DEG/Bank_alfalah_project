import cv2
from ultralytics import YOLO
from collections import defaultdict
import csv
from datetime import datetime
from datetime import datetime, timezone, timedelta
import httpx
import requests

branchId = 1
url = f"http://13.126.160.174:5000/suspicious?branchId={branchId}"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU"
headers = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true"
}
session = httpx.Client()

def search_suspicious(model1, model2, video_path):
    model_1 = model1
    model_2 = model2
    person = False
    # Load video
    video_path = video_path
    cap = cv2.VideoCapture(video_path)
    count = 0

    # Get frames per second of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Resize frame for faster processing (adjust resolution as needed)
            frame = cv2.resize(frame, (1300, 800))
            if person == False:
                person = track(model_1, frame)

            elif person == True:
                results2 = model_2.track(source=frame, show=True, project='./result', tracker="bytetrack.yaml", conf=0.4)
                for result2 in results2:
                    class_name2 = result2.boxes.cls.tolist()
                    conf_name2 = result2.boxes.conf.tolist()
                    res2 = dict(zip(class_name2, conf_name2))
                    print(res2)
                    if 0.0 in class_name2 and res2[0.0] >= 0.7:
                        print("ATM Card is Detected")
                        print("Person is non-suspicious act")
                        count = 0
                        person = False
                        timestamp = Time()                                
                        status = "NORMAL"
                        data = data_preparation(status,timestamp)
                        response = requests.post(url, json=data, headers=headers)
                        print("Endpoint response status code:", response.status_code)
                        # return "ATM Card is Detected"
                    elif count >= 1000:
                        count = 0
                        person = False
                        print('Person is doing some suspicious act')
                        timestamp = Time()                                
                        status = "SUSPICIOUS"
                        data = data_preparation(status,timestamp)
                        response = requests.post(url, json=data, headers=headers)
                        print("Endpoint response status code:", response.status_code)
                    else:
                        print(count)
                        count +=1    
            else:
                pass
        else:
            print(f"Working with frame empty frame, no frame detected")
            break

    cap.release()
    cv2.destroyAllWindows()

def track(model_1, frame):
    results = model_1.track(source=frame, show=False, project='./result', tracker="bytetrack.yaml", conf=0.4)
    for result in results:
        class_name = result.boxes.cls.tolist()
        conf_name = result.boxes.conf.tolist()
        res = dict(zip(class_name, conf_name))
    # res = [{cls: conf} for result in results for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())]
        if len(res) != 0:
            person = True
            print(res)
            return person
        else:
            pass
def Time():
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
    return formatted_date

def data_preparation(status,timestamp):
    data = {
    "status": status,
    "timestamp": timestamp
    }
    return data

if __name__ == '__main__':
    model1 = YOLO('track.pt')
    model2 = YOLO('ATM_Card.pt')
    # Load video
    video_path = 'test2.mp4'
    search_suspicious(model1, model2, video_path)