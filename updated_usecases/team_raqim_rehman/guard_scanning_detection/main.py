import math
import cv2
from ultralytics import YOLO
from datetime import datetime
import time
from datetime import datetime, timezone, timedelta
import httpx
import requests
                
branchId = 2
url = f"http://13.126.160.174:5000/scanning?branchId={branchId}"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtZWhyQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwODU3NjMyMCwiZXhwIjoxNzM0ODQxOTIwfQ.tb5RjpQe0tEfBbXuPmXLrHHAccSFJqlXga4SAxE56sU"
headers = {
    "Content-Type": "application/json",
    "Authorization": JWT_TOKEN,
    "X-SERVER-TO-SERVER": "true"
}
session = httpx.Client()

def calculate_centroid(box1):
    # Calculate the centroid of bounding box
    x_min, y_min, x_max, y_max = box1
    centroid_x = (x_min + x_max) / 2
    centroid_y = (y_min + y_max) / 2
    centroid = [centroid_x,centroid_y]
    return centroid

def pixels_to_inches_cv2(pixels, width_pixels, width_inches=None):
    """
    Convert pixels to inches based on the width of the image in pixels and inches (optional).
    If width_inches is not provided, it calculates the PPI based on the image's width in pixels.
    """
    if width_inches is None:
        # Calculate the PPI based on the image's width in pixels
        ppi = width_pixels / pixels
    else:
        # Calculate the PPI based on the provided width in inches
        ppi = width_pixels / width_inches
    inches = pixels / ppi
    return inches

def calculate_distance(centroid1, centroid2):
    # Calculate distance between two centroids
    pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
    return pixel_distance 

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

def guard_scanning_check(model_path, video_path, target_fps):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    count = 0
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / target_fps)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"Working with frame {frame_no}, no frame detected")
            break
        else:
            results = model.track(source=frame, show=True, project='./result', tracker="bytetrack.yaml", conf=0.4)
            for result in results:
                if result.boxes.id is not None:
                    trk_ids = result.boxes.id.int().cpu().tolist()
                    class_name = result.boxes.cls.tolist()
                    conf_name = result.boxes.conf.tolist()
                    boxes = result.boxes.xyxy.cpu().tolist()
                    cls_conf = res = dict(zip(class_name, conf_name))
                    res = dict(zip(class_name, boxes))
                    keys = list(res)
                    height, width, _ = frame.shape
                    if len(class_name) >= 2:
                        # if person=1.0 and scanner=2.0 classes are present
                        if 1.0 in class_name and cls_conf[1.0] >= 0.3 and 2.0 in class_name and cls_conf[2.0] >= 0.3:
                            box1=res[1.0]
                            box2=res[2.0]
                            centroid1 = calculate_centroid(box1)
                            centroid2 = calculate_centroid(box2)
                            print("Centroid:", centroid1)
                            print("Centroid:", centroid2)
                            distance_pixels = calculate_distance(centroid1, centroid2)
                            print('distance = ',distance_pixels, 'inches')
                            distance_inches = pixels_to_inches_cv2(distance_pixels, width)
                            print('distance = ',distance_inches, 'inches')
                            if distance_inches <= 125:
                                count = 0
                                timestamp = Time()                                
                                status = "SCANNED"
                                data = data_preparation(status,timestamp)
                                response = requests.post(url, json=data, headers=headers)
                                print("Endpoint response status code:", response.status_code)
                                print ('distance = ',distance_inches, 'Guard is Scanning')
                            # count is time in mili second to search for scanning, count can be change as per requirements     
                            elif count >= 1000:
                                timestamp = Time()
                                status = "NOT_SCANNED"
                                data = data_preparation(status,timestamp)
                                response = requests.post(url, json=data, headers=headers)
                                print("Endpoint response status code:", response.status_code)
                                print('Guard is not Scanning') 
                            else:
                                print(count)
                                count +=1
                        else:
                            pass
                    else:
                        pass 
            # Increment frame counter
        frame_no += 1
    # Release video capture
    cap.release()

if __name__ == '__main__':
    model_path = 'best (5).pt'
    video_path = "yt5s.io-Airport security goes too far!-(1080p).mp4"
    target_fps = 8
    print(guard_scanning_check(model_path, video_path, target_fps))


'''
Class, label
0, Guard
1, Person
2, Scanner
'''