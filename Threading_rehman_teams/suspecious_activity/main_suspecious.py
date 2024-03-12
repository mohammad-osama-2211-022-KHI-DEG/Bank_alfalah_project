import datetime
from ultralytics import YOLO
import logging
import os
from dotenv import load_dotenv
def setup_logging(log_filename):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
def load_model(model_path):
    """
    Load the YOLOv8 model.
    """
    return YOLO(model_path)
def load_video(video_path):
    """
    Load video from the specified path.
    """
    return cv2.VideoCapture(video_path)
def get_fps(video_capture):
    """
    Get the current frames per second (FPS) of the video capture object.
    """
    return video_capture.get(cv2.CAP_PROP_FPS)
def process_frame(model, frame):
    """
    Process the frame using the specified YOLO model.
    """
    return model(frame)
def detect_suspicious_conditions(class_ids):
    """
    Detect suspicious conditions based on detected class IDs.
    """
    suspicious_conditions = []
    if 4 in class_ids:
        suspicious_conditions.append('Helmet detected')
    num_persons = (class_ids == 5).sum()
    if num_persons > 2:
        suspicious_conditions.append('More than 2 persons detected')
    return suspicious_conditions
def check_person_duration(num_persons, first_person_time):
    """
    Check if one person is standing for more than 5 minutes.
    """
    if num_persons == 1 and first_person_time is None:
        return datetime.datetime.now()
    elif num_persons != 1:
        return None
    else:
        elapsed_time = datetime.datetime.now() - first_person_time
        if elapsed_time.total_seconds() > 1:  # 5 minutes in seconds
            return None
        return first_person_time
def main():
    # Load environment variables from .env file
    load_dotenv()
    # Constants
    LOG_FILENAME = os.getenv("LOG_FILENAME")
    VIDEO_PATH = os.getenv("VIDEO_PATH")
    MODEL_PATH = os.getenv("MODEL_PATH")
    # Set up logging
    setup_logging(LOG_FILENAME)
    # Load the YOLOv8 model
    model = load_model(MODEL_PATH)
    # Load video
    cap = load_video(VIDEO_PATH)
    # Get current FPS
    fps = get_fps(cap)
    print("Current FPS:", fps)
    # Initialize variables for tracking suspicious activity
    first_person_time = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Skip frames if not the 10th frame
        if frame_count % 1 != 0:
            logging.info(f"10th frame : {frame_count % 10}")
            continue
        #frame = cv2.resize(frame, (640, 480))
        results = process_frame(model, frame)
        # Get class IDs from the detected boxes
        class_ids = results[0].boxes.cls.numpy()
        # Detect suspicious conditions
        suspicious_conditions = detect_suspicious_conditions(class_ids)
        for condition in suspicious_conditions:
            print('Suspicious:', condition)
            logging.info(f"Suspicious: {condition}")
        # Check if one person is standing for more than 5 minutes
        first_person_time = check_person_duration((class_ids == 5).sum(), first_person_time)
        # Display the processed frame
        cv2.imshow('Branch', results[0].plot())
        #cv2.imshow('Branch_frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()