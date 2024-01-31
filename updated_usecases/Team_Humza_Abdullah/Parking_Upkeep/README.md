# Parking Monitoring System
This project is a Parking Monitoring System that uses YOLOv5 to detect cars in parking spaces and provides information about available parking spaces, occupied spaces, and wrong parking. The system also sends notifications in case of wrong parking.

## Prerequisites
Before running the code, make sure you have the following installed:

Python 3.x
OpenCV (pip install opencv-python)
PyTorch (pip install torch)
Ultralytics YOLOv5 (pip install yolov5)
Requests (pip install requests)
### Setup
Clone the repository:

Create a virtual environment (optional but recommended):

#### bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

### Usage
Open the terminal and navigate to the project directory.

Run the Parking Monitoring System:

### bash
Copy code
python parking_post_data_notification.py
The system will start processing the video feed, detecting cars, and providing information about parking spaces.

### Configuration
You can modify the parking_post_data_notification.py file to change configuration parameters, such as video source, endpoint URLs, and authorization headers.