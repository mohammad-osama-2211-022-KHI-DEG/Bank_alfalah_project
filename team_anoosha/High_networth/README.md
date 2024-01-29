# High Networth

## Overview
This Python script, `High_networth.py`, is designed for high-net-worth individual recognition in video streams. It utilizes face recognition techniques to identify individuals and sends alerts for recognized persons.

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
* `VIDEO_PATH`: Path to the input video.
* `ENCODINGS_FILE`: File path to store face encodings.
* `OUTPUT_VIDEO`: Output video path with recognized individuals marked.
* `TOLERANCE`: Face recognition tolerance level.
* `ALERT_API_URL`: API endpoint for sending alerts.
* `API_HEADERS`: Headers for API request, including authorization.

## Logging
Logs are generated at INFO level, providing information about the script's execution.

## Recognized Individuals
The script maintains a set of recognized individuals to trigger alerts only once for each person.

## Functions
`load_encodings(file_path)`: Load face encodings from the specified file.
`generate_alert(name)`: Generate and send an alert for the recognized individual.
`recognize_faces(frame, face_detector, data)`: Recognize faces in a given frame and trigger alerts.

## Usage
1. Ensure Python dependencies are installed: `pip install -r requirements.txt`.
2. Run `manipulate_img.py` to augment images in '`total_img`'.
3. Run `get_emb.py` to generate face encodings.
4. Execute `High_networth.py` to start face recognition on the specified video.

## Alerts
The project is configured to send alerts when a recognized individual is detected in the video stream. Alerts are sent to a specified API endpoint with relevant information such as the detected individual's name and timestamp.
