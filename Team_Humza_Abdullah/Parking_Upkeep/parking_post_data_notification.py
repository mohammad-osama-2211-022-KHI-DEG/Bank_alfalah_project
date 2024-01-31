import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import datetime
import requests
from datetime import datetime, timezone, timedelta

# Define constants
TAG = "parking"
ENDPOINT_URL = "http://13.235.71.140:5000/parking"
NOTIFICATION_URL = "http://13.235.71.140:5000/notification"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtdWhhbW1hZG9zYW1hLmhxQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwNjU5NzA5NCwiZXhwIjoxNzA2ODk3MDk0fQ.b8Fnd3T5Egmd_r5vyi-7u1eF175JJAjsYD8uy_1-f00",
    "X-SERVER-TO-SERVER": "true"
}

# Initialize variables to store previous values
previous_total_spaces = None
previous_occupied_spaces = None
previous_free_spaces = None
previous_wrong_parking = None

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def is_horizontal_parking(x1, y1, x2, y2):
    return (x2 - x1) > (y2 - y1)

def get_formatted_date():
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
    return formatted_date

def send_notification(message, country, city, branch, usecase, timestamp):
    data = {
        "timestamp": timestamp,
        "message": message,
        "country": country,
        "city": city,
        "branch": branch,
        "usecase": usecase
    }
    response = requests.post(NOTIFICATION_URL, json=data, headers=HEADERS)

    if response.status_code == 200:
        print(f"Notification sent successfully at {timestamp}")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}")

def save_parking_data(total_spaces, occupied_spaces, free_spaces, wrong_parking):
    global previous_total_spaces, previous_occupied_spaces, previous_free_spaces, previous_wrong_parking

    # Check if there's any change in the parking status
    if (
        total_spaces != previous_total_spaces
        or occupied_spaces != previous_occupied_spaces
        or free_spaces != previous_free_spaces
        or wrong_parking != previous_wrong_parking
    ):
        # Get the current timestamp
        timestamp = get_formatted_date()

        # Prepare the data to be stored in the document
        data = {
            "country": "pakistan",
            "branch": "clifton",
            "city": "karachi",
            "timestamp": timestamp,
            "noOfCarsParked": occupied_spaces,
            "vacantSpaces": free_spaces,
            "wrongParked": wrong_parking,
            "parkingSpaces": total_spaces
        }

        # Replace {TAG} with the actual tag
        endpoint_url = ENDPOINT_URL.format(TAG=TAG)

        # Send POST request to the endpoint
        response = requests.post(endpoint_url, json=data, headers=HEADERS)

        if response.status_code == 200:
            print(f"Data posted successfully at {timestamp}")
        else:
            print(f"Failed to post data. Status code: {response.status_code}")

        # Check if wrong parking is detected
        if wrong_parking > 0:
            # Send notification
            send_notification("Wrong Parking Detected", "parkistan", "karachi", "clifton", "parking", timestamp)

        # Update the previous values for the next iteration
        previous_total_spaces = total_spaces
        previous_occupied_spaces = occupied_spaces
        previous_free_spaces = free_spaces
        previous_wrong_parking = wrong_parking


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def is_horizontal_parking(x1, y1, x2, y2):
    return (x2 - x1) > (y2 - y1)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('parking.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


area9 = [(675, 1), (925, 1), (925, 185), (670, 185)]
area8 = [(530, 1), (670, 1), (670, 185), (527, 185)] #6
area7 = [(402, 1), (528, 1), (525, 185), (396, 185)]#8
area6 = [(265, 1), (400, 1), (388, 185), (265, 185)] #7
area5 = [(77, 1), (260, 1), (260, 185), (75, 185)]
area4 = [(769, 280), (930, 293), (930, 497), (788, 496)] #4
area3 = [(527, 253), (765, 280), (785, 498), (457, 496)]
area2 = [(300, 225), (525, 250), (448, 498), (114, 470)] #2
area1 = [(76, 194), (296, 221), (108, 468), (77, 463)]


while True:
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (1020, 500))
    results = model(frame)

    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []

    wrong_parking = 0
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if 'car' in c:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check if the car is within the frame
            if 0 <= cx < 1020 and 0 <= cy < 500:
                # Check if the car is parked incorrectly
                if (
                    (cv2.pointPolygonTest(np.array(area9, np.int32), ((cx, cy)), False) < 0) or
                    not is_horizontal_parking(x1, y1, x2, y2)
                ):
                    cv2.putText(frame, "Wrong Parking", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                    wrong_parking += 1

      # Draw parking space and other visualizations as before
                results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                list1.append(c)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)




        results2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
        if results2>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list2.append(c)
            
        results3=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
        if results3>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list3.append(c)   
        results4=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
        if results4>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list4.append(c)  
        results5=cv2.pointPolygonTest(np.array(area5,np.int32),((cx,cy)),False)
        if results5>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list5.append(c)  
        results6=cv2.pointPolygonTest(np.array(area6,np.int32),((cx,cy)),False)
        if results6>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list6.append(c)  
        results7=cv2.pointPolygonTest(np.array(area7,np.int32),((cx,cy)),False)
        if results7>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list7.append(c)   
        results8=cv2.pointPolygonTest(np.array(area8,np.int32),((cx,cy)),False)
        if results8>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list8.append(c)  
        results9=cv2.pointPolygonTest(np.array(area9,np.int32),((cx,cy)),False)
        if results9>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list9.append(c)  

    a1 = (len(list1))
    a2 = (len(list2))
    a3 = (len(list3))
    a4 = (len(list4))
    a5 = (len(list5))
    a6 = (len(list6))
    a7 = (len(list7))
    a8 = (len(list8))
    a9 = (len(list9))

    o = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9)
    space = (9 - o)

    print(space)


    if a1==1:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('1'),(141,321),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('1'),(141,321),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a2==1:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('2'),(331,343),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('2'),(331,343),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a3==1:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('3'),(624,373),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('3'),(624,373),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a4==1:
        cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('4'),(840,381),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('4'),(840,381),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a5==1:
        cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('5'),(128,61),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('5'),(128,61),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a6==1:
        cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('6'),(220,101),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('6'),(220,101),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
    if a7==1:
        cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('7'),(349,117),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('7'),(349,117),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a8==1:
        cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('8'),(500,142),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('8'),(500,142),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)  
    if a9==1:
        cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('9'),(668,155),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('9'),(668,155),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    # Display total available parking spaces
    total_spaces = 9
    occupied_spaces = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9
    free_spaces = total_spaces - occupied_spaces

    font_size = 1  

    cv2.putText(frame, f"Total Spaces: {total_spaces}", (23, 30), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)
    cv2.putText(frame, f"Occupied Spaces: {occupied_spaces}", (23, 60), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)
    cv2.putText(frame, f"Free Spaces: {free_spaces}", (23, 90), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)
    cv2.putText(frame, f"Wrong Parking: {wrong_parking}", (23, 120), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)

    # Call the function to save parking data
    save_parking_data(total_spaces, occupied_spaces, free_spaces, wrong_parking)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
