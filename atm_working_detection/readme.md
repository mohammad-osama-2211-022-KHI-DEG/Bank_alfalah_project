# ATM Functionality Detection

This repository contains a Python script for detecting ATM functionality in a video using YOLO (You Only Look Once) object detection. It checks for the presence of an ATM card, cash, and other components to determine if the ATM is working properly.

## Requirements

Make sure you have the required dependencies installed by running:

```bash
pip install -r requirements.txt


## Usage
To run the script, execute the following command:

'''bash
python3 main.py

This script uses the YOLO model located at atm_functionality_best.pt and processes the video specified in video_path. You can customize the video file and other parameters in the main.py script according to your needs.

## Configuration
- model_path: Path to the YOLO model file (atm_functionality_best.pt).
- video_path: Path to the input video file.
- target_fps: Target frames per second for processing the video.
- Output:
The script analyzes the video and sends the results to a specified endpoint (http://13.235.71.140:5000/atm-functionality) along with additional information such as working status, transaction counts, and the availability of complaint boxes and telephones.

## Notifications
If the ATM is not working, the script sends a notification to another endpoint (http://13.235.71.140:5000/notification) with details about the issue, including a timestamp, message, country, city, branch, and use case.

## Contact
For any inquiries or issues, please contact the repository owner.
