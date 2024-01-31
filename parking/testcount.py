import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('tesing.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area11 = [(76, 3), (231, 5), (78, 172), (76, 91)]
area10 = [(112, 203), (285, 230), (107, 455), (76, 439)]
area9 = [(535, 252), (463, 498), (786, 498), (761, 284)]
area8 = [(77, 180), (179, 201), (352, 18), (243, 11)]
area7 = [(350, 13), (471, 27), (361, 221), (183, 201)]
area6 = [(473, 32), (603, 46), (553, 243), (370, 223)]
area5 = [(609, 52), (722, 64), (761, 265), (561, 250)]
area4 = [(764, 282), (913, 291), (923, 492), (793, 490)]
area3 = [(726, 60), (774, 266), (915, 262), (836, 72)]
area2 = [(726, 60), (774, 266), (915, 262), (836, 72)]
area1 = [(297, 244), (525, 254), (453, 496), (110, 460)]



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
    list10 = []
    list11 = []
    
    wrongparking=0
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if 'car' in c:
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            # Check if the car is parked outside of the defined areas
            if (cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area3, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area4, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area5, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area6, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area7, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area8, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area9, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area10, np.int32), ((cx, cy)), False) < 0 or
                cv2.pointPolygonTest(np.array(area11, np.int32), ((cx, cy)), False) < 0 or
                # Check for horizontal parking
                x2 - x1 > y2 - y1
                ):
                # Check if the car ID has already triggered a wrong parking alert in this frame
                
                
                
                cv2.putText(frame, "Wrong Parking", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                wrongparking += 1       
                # Draw parking space and other visualizations as before
                results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                list1.append(c)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        # Repeat for other areas


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
        results10=cv2.pointPolygonTest(np.array(area10,np.int32),((cx,cy)),False)
        if results10>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list10.append(c)     
        results11=cv2.pointPolygonTest(np.array(area11,np.int32),((cx,cy)),False)
        if results11>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list11.append(c)    

              
            
    a1=(len(list1))
    a2=(len(list2))       
    a3=(len(list3))    
    a4=(len(list4))
    a5=(len(list5))
    a6=(len(list6)) 
    a7=(len(list7))
    a8=(len(list8)) 
    a9=(len(list9))
    a10=(len(list10))
    a11=(len(list11))
    # a12=(len(list12))
    o=(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11)
    space=(10-o)
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
    if a10==1:
        cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('10'),(822,170),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('10'),(822,170),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a11==1:
        cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('11'),(822,170),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('11'),(697,377),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    # # ... (code for other areas)

    # Display total available parking spaces
    total_spaces = 10
    occupied_spaces = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 
    free_spaces = total_spaces - occupied_spaces

    font_size = 1  # You can change this value to adjust the font size

    cv2.putText(frame, f"Total Available Spaces: {total_spaces}", (23, 30), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)
    cv2.putText(frame, f"Occupied Spaces: {occupied_spaces}", (23, 60), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)
    cv2.putText(frame, f"Free Spaces: {free_spaces}", (23, 90), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)
    cv2.putText(frame, f"Wrong Parking: {wrongparking}", (23, 120), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)

    # cv2.putText(frame, f"Wrong parking: {free_spaces}", (23, 90), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), 2)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()