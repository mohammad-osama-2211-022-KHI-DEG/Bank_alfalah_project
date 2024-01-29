## Usage
1. Install the required dependencies using the provided requirements.txt file:
```
pip install -r requirements.txt
```

2. Run the script:
```
customer_waiting_time.py
``` 

## Configuration
* `CONFIDENCE_THRESHOLD`: Confidence threshold for face detection.
* `FACE_RECOGNITION_TOLERANCE`: Tolerance level for face recognition.
* `TIME_THRESHOLD`: Threshold to reset total duration after a certain time.
* `ID_DISAPPEAR_THRESHOLD`: Threshold for face ID disappearance.
* `VIDEO_PATH`: Path to the input video.
* `OUTPUT_PATH`: Path for the output video.
* `JWT_TOKEN`: JSON Web Token for server authorization.
* `API_BASE_URL`: Base URL for the server API.
* `HEADERS`: HTTP headers for server requests.

## Logging
The script logs information, warnings, and errors to a file. You can find the logs in the output/logs/mart.log file.

## Server Communication
The script communicates with a server to record face-related data, such as entry and exit timestamps. The server's base URL and API endpoints are configured in the API_BASE_URL and HEADERS variables.

## Note
Ensure that the video file is accessible and the required permissions are set.
Adjust the configuration constants based on your specific requirements.
This script is designed to work with a specific server API, and modifications may be necessary for integration with a different backend.