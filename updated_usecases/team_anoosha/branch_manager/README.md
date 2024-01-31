# Branch Manager

## Overview
This directory contains Python scripts for a `branch_manager` using MTCNN and face_recognition libraries. The system detects faces in a video stream, recognizes them, and logs their appearance duration. Additionally, there are scripts to generate face encodings for recognition and manipulate images for data augmentation.

## Scripts

1. `branch_manager.py`
* Main script for face detection and recognition in a video stream.
* Monitors face appearances, calculates durations, and sends data to the server on certain conditions.
* Uses MTCNN and face_recognition libraries for face detection and recognition.

2. `get_emb.py`
* Generates face encodings for known faces in the '`augmented_img`' folder.
* Utilizes the face_recognition library to extract encodings and saves them in a pickle file.

3. `manipulate_img.py`
* Performs image processing tasks for data augmentation.
* Subdirectories in '`total_img`' are processed to create a new set of augmented images in the 'augmented_img' folder.

## Usage
1. Ensure Python dependencies are installed: `pip install -r requirements.txt`.
2. Run `manipulate_img.py` to augment images in '`total_img`'.
3. Run `get_emb.py` to generate face encodings.
4. Execute `branch_manager.py` to start face recognition on the specified video.

## Logging:
* The application logs details to the '`output/logs/nov.log`' file.
* Log entries include timestamps, log levels, and informative messages.

## Configuration
* `FACE_RECOGNITION_TOLERANCE`: Tolerance level for face recognition comparisons.
* `REAPPEARANCE_THRESHOLD`: Time threshold for detected face reappearances.
* `ENCODINGS_FILE`: File path for storing face encodings.
* `VIDEO_PATH`: Input video file path for processing.
* `OUTPUT_VIDEO_PATH`: Output video file path with annotations.
* `FOURCC`: FourCC code for video codec selection.
* `JWT_TOKEN`: JSON Web Token for server authentication.
* `API_BASE_URL`: Base URL for the server API.
* `HEADERS`: HTTP headers for server requests.

## Note
The system logs alerts for faces reappearing after a certain threshold.
Make sure to configure the API details for server communication.
