import cv2
from ultralytics import YOLO
from datetime import datetime
import time
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase with your credentials and database URL
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://test-3b45c-default-rtdb.firebaseio.com/"})
ref2 = db.reference('/ATM Working Detection')
# Function for the atm-card and cash detection
def detect_atm_usage(model_path, video_path, target_fps):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / target_fps)
    counter = 0
    count = 0
    atm_detected = False

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print(f"Working with frame {frame_no}, no frame detected")
            break
        else:
            # **ATM card and note detection logic**
            results = model.track(source=frame, show=True, project='./result', tracker="bytetrack.yaml", conf=0.4)
            a = None
            for result in results:
                labels = result.boxes.data.cpu().numpy()
                if len(labels) != 0 and atm_detected == False:
                    a = labels[0]
                    if atm_detected == False and a[-1] == 0 and a[-2] > 0.4:
                        print("ATM Card is Detected")
                        atm_detected = True
                        count = 0
                    else:
                        pass
                elif len(labels) != 0 and atm_detected == True:
                    a = labels[0]
                    if a[-1] == 1 and a[-2] > 0.8:
                        # print(labels)
                        print("Cash is Detected")
                        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_data = {
                            "Alert": "ATM is Working",
                            "timestamp": time_now
                        }
                        # Push data to the database
                        ref2.push(new_data)
                        atm_detected = False
                        # return "ATM is Working"
                        # break
                    elif count >= 1000:
                        print("ATM detection timeout reached. Exiting...")
                        counter += 1
                        push_data_to_database(counter)
                        atm_detected = False
                        # return "ATM is not Working"
                    else:
                        count +=1
                elif atm_detected == True:
                    if count >= 1000:
                        print("ATM detection timeout reached. Exiting...")
                        counter += 1
                        push_data_to_database(counter)
                        atm_detected = False
                    else:
                        count +=1
                else:
                    pass
            # Increment frame counter
        frame_no += 1
    # Release video capture
    cap.release()
# ... (your push_data_to_database function)
def push_data_to_database(counter):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Counting for 6 cash not detected
    if counter >= 6:
        # Your data to be added
        new_data = {
            "Alert": "ATM is not Working",
            "timestamp": time_now
        }
        # Push data to the database
        
        ref2.push(new_data)
        counter = 0

# Example usage (unchanged):
model_path = 'atm_working.pt'
video_path = "YouCut_20240114_230531505.mp4"
target_fps = 8
detect_atm_usage(model_path, video_path, target_fps)

