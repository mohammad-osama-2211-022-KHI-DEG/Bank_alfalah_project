import os
import cv2


OUTPUT_VIDEO_PATH = 'output/output1.avi'
FOURCC = cv2.VideoWriter_fourcc(*'XVID')

username = 'usama.xloop'
password = 'Xloop12345'
ip_address = '192.168.6.13'
url = f"rtsp://{username}:{password}@{ip_address}:554/cam/realmonitor?channel=1&subtype=1"
url2 = "rtsp://192.168.100.7:1578/video"
# Open the video stream
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
print(f'cap:{cap}')
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, FOURCC, fps, (frame_width, frame_height))
print(fps)

while True:
    
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(frame)
    cv2.imshow('Live Stream', frame)
# Release the video stream and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
