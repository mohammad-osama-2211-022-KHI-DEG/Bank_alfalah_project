import cv2
from ultralytics import YOLO
from datetime import datetime
import time
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase with credentials and database URL
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://test-3b45c-default-rtdb.firebaseio.com/"})
ref = db.reference('/Complain-box & Telephone')
ref2 = db.reference('/ATM Working Detection')

# ATM functionality check 
def detect_atm_usage(model_path, video_path, target_fps):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / target_fps)
    counter = 0
    count = 0
    atm_detected = False
    total_working_count=0
    total_notworking_count=0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print(f"Working with frame {frame_no}, no frame detected")
            break
        else:
            results = model.track(source=frame, show=True, project='./result', tracker="bytetrack.yaml", conf=0.4)
            a = None
            for result in results:
                class_name = result.boxes.cls.tolist()
                conf_name = result.boxes.conf.tolist()
                res = dict(zip(class_name, conf_name))
                # check for complain-box and telephone
                complainbox_telephone(class_name)
                if len(res) != 0 and atm_detected == False:
                    if atm_detected == False and 0.0 in class_name and res[0.0] >= 0.4:
                        print("ATM Card is Detected")
                        atm_detected = True
                        count = 0
                    else:
                        pass
                elif len(res) != 0 and atm_detected == True:
                    if 1.0 in class_name and res[1.0] >= 0.8:
                        print("Cash is Detected")
                        # pushing data to db
                        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        alert = "ATM is Working"
                        push_data_to_database(alert,time_now,ref2)
                        total_working_count +=1
                        # Push data to the database
                        db.reference("/Total successful transcation").set(total_working_count)
                        atm_detected = False
                    elif count >= 1000:
                        print("ATM detection timeout reached. Exiting...")
                        counter += 1
                        total_notworking_count +=1
                        # Push data to the database
                        db.reference("/Total Unsuccessful transcation").set(total_notworking_count)
                        # Counting for 6 cash not detected
                        if counter >= 6:
                            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            alert = "ATM is not Working"
                            push_data_to_database(alert,time_now,ref2)
                            counter = 0
                        atm_detected = False
                    else:
                        count +=1
                elif atm_detected == True:
                    if count >= 1000:
                        print("ATM detection timeout reached. Exiting...")
                        counter += 1
                        total_notworking_count +=1
                        # Push data to the database
                        db.reference("/Total Unsuccessful transcation").set(total_notworking_count)
                        # Counting for 6 cash not detected
                        if counter >= 6:
                            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            alert = "ATM is not Working"
                            push_data_to_database(alert,time_now,ref2)
                            counter = 0
                        atm_detected = False
                    else:
                        count +=1
                else:
                    pass
            else:
                pass

            # Increment frame counter
        frame_no += 1
    # Release video capture
    cap.release()

# Check for complain-box and telephone
def complainbox_telephone(class_name):
    # Complain-box & Telephone are present
    if 2.0 in class_name and 3.0 in class_name:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = "Complain box & telephone found"
        # Push data to the database
        push_data_to_database(alert,time_now,ref)
    # Complain-box is present & Telephone is not present
    elif 2.0 in class_name and 3.0 not in class_name:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = "Complain box found & telephone not found"
        # Push data to the database
        push_data_to_database(alert,time_now,ref)
    # Complain-box is not present & Telephone is present
    elif 2.0 not in class_name and 3.0 in class_name:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = "Complain box not found & telephone found"
        # Push data to the database
        push_data_to_database(alert,time_now,ref)
    # Complain-box & Telephone are not present
    elif 2.0 not in class_name and 3.0 not in class_name:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = "Complain box & telephone not found"
        # Push data to the database
        push_data_to_database(alert,time_now,ref)    
# Pushing data to database
def push_data_to_database(alert,time_now,reff):
    new_data = {
        "Alert": alert,
        "timestamp": time_now
    }
    reff.push(new_data)

if __name__ == '__main__':
    model_path = 'atm_functionality_best.pt'
    video_path = "YouCut_20240114_230531505.mp4"
    target_fps = 8
    detect_atm_usage(model_path, video_path, target_fps)
