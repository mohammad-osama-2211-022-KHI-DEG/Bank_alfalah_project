# Aggressive Customer Detection 

## Overview

This Python script implements a Aggressive  Customer Detection  using computer vision and machine learning techniques. The system analyzes a video feed, identifies faces using the MTCNN (Multi-task Cascaded Convolutional Networks) face detection model, and classifies emotions using a pre-trained deep learning model. It then tracks happy customers, counts unique happy faces, and sends notifications and emotion data to a remote server.

## Requirements

Ensure you have the following dependencies installed:

- **OpenCV (`cv2`)**: Computer vision library for image and video processing.
- **MTCNN (`mtcnn`)**: Face detection library.
- **face_recognition**: Facial recognition library.
- **NumPy (`numpy`)**: Library for numerical operations.
- **Keras (`keras`)**: Deep learning library for building and training models.
- **Requests (`requests`)**: HTTP library for sending requests.

You can install these dependencies using:

```bash
pip install opencv-python mtcnn face_recognition numpy keras requests
```
## Configuration

Before running the script, make sure to set the following constants in the code:

- **DISTANCE_THRESHOLD**: Face recognition distance threshold.
- **WINDOW_WIDTH** and **WINDOW_HEIGHT**: Size of the display window.
- **VIDEO_FILE_PATH**: Path to the video file for processing.
- **EMOTION_DETECTION_MODEL_PATH**: Path to the pre-trained emotion detection model (in HDF5 format).
- **SERVER_URL**: URL of the remote server for sending notifications and emotion data.

Ensure that you replace the `Authorization` header in the `get_request_headers` function with your own **JWT token**.
## Usage

Run the script by executing the following command in your terminal or command prompt:

```bash
python3 main.py
```
The script will process the video feed, detect faces, classify Aggresive emotions, and display the results in a window.

Emotion data and notifications will be sent to the specified server.

### Server Integration
The script is designed to interact with a server for storing emotion data and sending notifications. Ensure that the server is configured to handle incoming data and notifications. The server endpoints used are **/emotion/aggressive** for emotion data and **/notification** for sending notifications.

### Notifications
The system sends notifications when the number of unique happy customers reaches a certain threshold. Adjust the conditions in the script according to your preferences.

### Exit
Press 'q' to exit the video feed window and close the script.

