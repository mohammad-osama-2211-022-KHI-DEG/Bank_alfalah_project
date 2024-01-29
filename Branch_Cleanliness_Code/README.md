# Branch Cleanliness Monitoring System
This Python script implements a Cleanliness Monitoring System using YOLO (You Only Look Once) object detection for detecting tables and objects within a given video of any bank branch. The system assesses the cleanliness status of tables based on the detected objects on them.

### Prerequisites
#### Libraries:
os: Operating system interfaces
ultralytics: YOLO (You Only Look Once) framework for object detection
cv2: OpenCV library for computer vision
requests: HTTP library for making requests
datetime, timezone, timedelta: Date and time manipulation
Requirements

### Video File:

Provide the input video file (final_testing.mp4) in the same directory as the script.

### YOLO Models:

Store YOLO models (best.pt) for tables and objects in tables_weights and objects_weights folders, respectively.
Python Environment (Optional):

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
Functionality

### Object Detection:

Utilizes YOLO models (best.pt) for detecting tables and objects within the video frames.


### Status Assessment:

Determines the cleanliness status of each detected table based on the number and type of objects on it.

### Data Posting:

Posts cleanliness data to a specified endpoint, including timestamp, cleanliness state, country, branch, city, and more.

### Video Output:

Generates an output video (final_testing_out.mp4) with annotated YOLO detections, including table status.
Usage
### Activate Virtual Environment (Optional):

Activate the virtual environment if created.
bash
Copy code
source venv/bin/activate  # On Windows, use "venv\Scripts\activate"


### Run the Script:

Execute the script, and the system will process the video, perform object detection, assess cleanliness, and generate the output video.
bash
Copy code
python best_practice.py


### Overall Summary:

1: The script includes methods for drawing YOLO detections, calculating cleanliness status, building payload data, and posting data to an endpoint.

2: The cleanliness status is determined based on the number and type of objects detected on each table.

3: The script utilizes YOLO to annotate video frames, indicating the status of each detected table.

4: The system posts cleanliness data to a specified endpoint, providing real-time information about the cleanliness status.

5: The output video visually represents the detected tables, annotated with cleanliness status and object detections.
