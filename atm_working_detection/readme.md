# ATM Functionality Detection

This repository contains a Python script for detecting ATM functionality in a video using YOLO (You Only Look Once) object detection. It checks for the presence of an ATM card, cash, and other components to determine if the ATM is working properly.

## Requirements

Make sure you have the required dependencies installed by running:


> pip install -r requirements.txt


## Usage
To run the script, execute the following command:

> python3 main.py

This script uses the YOLO model located at atm_functionality_best.pt and processes the video specified in video_path. You can customize the video file and other parameters in the main.py script according to your needs.

## Configuration
- model_path: Path to the YOLO model file (atm_functionality_best.pt).
- video_path: Path to the input video file.
- target_fps: Target frames per second for processing the video.
- Output:
The script analyzes the video and sends the results to a specified endpoint (http://13.235.71.140:5000/atm-functionality) along with additional information such as working status, transaction counts, and the availability of complaint boxes and telephones.

## About Model 
We are using YOLOv8 from Ultralytics. Ultralytics YOLOv8 is not just another object detection model; it's a versatile framework designed to cover the entire lifecycle of machine learning models—from data ingestion and model training to validation, deployment, and real-world tracking.

## Working Logic
- Upon a person's arrival at the ATM, the model initiates the detection process for an ATM card.
- Upon successfully detecting an ATM card, the variable atm_detected is set to True, triggering a 100-second timer to search for cash in the frame.
- If cash is detected within the 100-second timeframe, the total_working_count is incremented, and workingStatus is set to True.
- If no cash is detected within the allotted time, the counter variable is incremented.
- When the counter reaches 2, indicating a lack of cash detection in two consecutive attempts, workingStatus is set to False.
- In such cases, the total_notworking_count is incremented, and the Alert() function is triggered to send a notification with the message "atm_is_notworking".

Additionally, it's worth noting that the presence of complaintBoxAvailable and telephoneAvailable is checked in every frame. However, their status is reported whenever the workingStatus is triggered, either True or False. This ensures that the availability of complaint boxes and telephones is communicated only when there is a change in the working status of the ATM.

## Notifications
If the ATM is not working, the script sends a notification to another endpoint (http://13.235.71.140:5000/notification) with details about the issue, including a timestamp, message, country, city, branch, and use case.

## Contact
For any inquiries or issues, please contact the repository owner.
