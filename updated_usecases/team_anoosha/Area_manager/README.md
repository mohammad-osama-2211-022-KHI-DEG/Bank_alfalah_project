# Area Manager

## Overview
This Python script, `area_manager.py`, is designed for area_manager recognition in video streams. It utilizes face recognition techniques to identify individuals and sends alerts for recognized persons.

## Dependencies
Make sure you have the following dependencies installed:
* OpenCV
* MTCNN
* face_recognition
* pickle
* logging
* datetime
* requests

## Constants
* `FILENAME`: Video file name.
* `TOLERANCE`: Face recognition tolerance level.
* `LOG_FILENAME`: Define log file address.
* `VIDEO_PATH`: Path to the input video.
* `OUTPUT_VIDEO`: Output video path with recognized individuals marked.
* `ENCODINGS_FILE`: File path to store face encodings.
* `ALERT_API_URL`: API endpoint for sending alerts.
* `NOTIFICATION_URL`: Notification endpoint for sending notification alerts.
* `JWT_TOKEN`: Authorization code.
* `API_HEADERS`: Headers for API request, including authorization.

## Logging
Logs are generated at DEBUG level, providing information about the script's execution.

## Recognized Individuals
The script maintains a set of recognized individuals to trigger alerts only once for each person.

## Functions
* `load_encodings(file_path)`: Load face encodings from the specified file.
* `generate_alert(name)`: Generate and send an alert for the recognized individual.
* `recognize_faces(frame, face_detector, data)`: Recognize faces in a given frame and trigger alerts.

## Usage
1. Ensure Python dependencies are installed: `pip install -r requirements.txt`.
2. Run `get_and_save_emb.py` to generate and save face encodings.
3. Execute `area_manager.py` to start face recognition on the specified video.

## Alerts
The project is configured to send alerts when a recognized individual is detected in the video stream. Alerts are sent to a specified API endpoint with relevant information such as the detected individual's name and timestamp.
