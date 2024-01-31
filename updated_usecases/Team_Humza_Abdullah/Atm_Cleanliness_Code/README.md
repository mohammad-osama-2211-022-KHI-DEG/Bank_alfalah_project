# Cleanliness Monitoring System with YOLO
## Overview
This Python script implements a Cleanliness Monitoring System using YOLO (You Only Look Once) object detection. The system analyzes a video feed to detect objects, assesses cleanliness levels, and posts relevant data to a remote endpoint. It incorporates real-time monitoring, annotation of video frames, and potential notification alerts for high mess levels.

### Prerequisites
#### Libraries:
os: Operating system interfaces
ultralytics: YOLO (You Only Look Once) framework for object detection
cv2: OpenCV library for computer vision
requests: HTTP library for making requests
datetime, timezone, timedelta: Date and time manipulation
Requirements

### Video File:

Provide the input video file (testing.mp4) or (alfa_messy_video.mp4) in the same directory as the script.

### YOLO Models:

Store YOLO models (best.pt) for atm cleanlinessin the same directory as the script.

### Create and activate a virtual environment (venv) for isolation:
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use "venv\Scripts\activate"

### Install Dependencies:

Install required Python packages:
bash
Copy code
pip install -r requirements.txt

## Features
Object Detection: Uses YOLO for real-time object detection in video frames.
ATM Cleanliness Monitoring: Calculates the level of mess based on detected objects.
Data Posting: Posts cleanliness data to a specified endpoint with timestamp and cleanliness status.
Notifications: Sends notifications for high mess levels.
## Usage
Execute the script by running python script_name.py.
Observe the video feed with real-time object counts, cleanliness status, and mess level.
Data is posted to the specified endpoint based on changes in cleanliness status or mess level.
Notifications are sent for high mess levels.
## Termination
The script can be terminated by pressing 'q' during execution.