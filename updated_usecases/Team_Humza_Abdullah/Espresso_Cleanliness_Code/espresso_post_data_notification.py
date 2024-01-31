from ultralytics import YOLO
import numpy as np
import cv2
import requests
from datetime import datetime, timezone, timedelta

def load_yolo_model(model_path):
    return YOLO(model_path)

def open_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    return cap

def create_video_writer(output_path, fps, width, height):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

def get_formatted_date():
    current_datetime = datetime.now(timezone(timedelta(hours=5)))
    formatted_date = current_datetime.strftime('%Y-%m-%dT%H:%M:%S') + str('+05:00')
    return formatted_date

def prepare_cleanliness_data(cleanliness_status, level_of_mess):
    return {
        "timestamp": get_formatted_date(),
        "levelOfMess": level_of_mess,
        "cleanlinessState": cleanliness_status,
        "country": "pakistan",
        "branch": "clifton",
        "city": "karachi"
    }

def send_data_to_cleanliness_endpoint(cleanliness_data):
    url = f"http://13.235.71.140:5000/cleanliness/espresso"
    try:
        response = requests.post(url, json=cleanliness_data, headers=headers)
        response.raise_for_status()  
        print(response)
        print(response.content)
        if response.status_code == 200:
            print("Data sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Data. Exception: {e}")

def send_alert_notification(message, country, city, branch, usecase, timestamp):
    notification_data = {
        "timestamp": timestamp,
        "message": message,
        "country": country,
        "city": city,
        "branch": branch,
        "usecase": usecase
    }
    alert_url = "http://13.235.71.140:5000/notification"
    try:
        response = requests.post(alert_url, json=notification_data, headers=headers)
        response.raise_for_status()
        if response.status_code == 200:
            print("Notification sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send notification. Exception: {e}")

def process_frame(frame, model, previous_cleanliness_status, previous_level_of_mess):
    # Make predictions on the current frame
    results = model(frame)

    # Extract predictions
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    # Determine cleanliness status and level of mess
    cleanliness_status = names_dict[np.argmax(probs)]
    level_of_mess = max(probs) * 100  # Convert probability to percentage

    # Check if cleanliness status or level of mess has changed
    if (cleanliness_status != previous_cleanliness_status) or (level_of_mess != previous_level_of_mess):
        # Prepare data for cleanliness API
        cleanliness_data = prepare_cleanliness_data(cleanliness_status, level_of_mess)

        # Post data to cleanliness endpoint
        send_data_to_cleanliness_endpoint(cleanliness_data)

        # Check if the level of mess exceeds 50% and send an alert
        if level_of_mess > 90:
            alert_message = "High level of mess detected!"
            send_alert_notification(alert_message, "pakistan", "karachi", "clifton", "cleanliness", cleanliness_data["timestamp"])

    return names_dict, probs, cleanliness_status, level_of_mess

def overlay_text_on_frame(frame, predicted_class, max_prob):
    # Overlay class names and probabilities on the frame
    overlay_text = f"Class: {predicted_class}, Prob: {max_prob:.2f}"
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def main(video_path, model_path, output_path):
    # Load the YOLO model
    model = load_yolo_model(model_path)

    # Open a video capture object
    cap = open_video_capture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter object to save the output video
    out = create_video_writer(output_path, fps, width, height)

    # Initialize variables for cleanliness status and level of mess
    previous_cleanliness_status = None
    previous_level_of_mess = None

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Process the frame
        names_dict, probs, previous_cleanliness_status, previous_level_of_mess = process_frame(frame, model, previous_cleanliness_status, previous_level_of_mess)

        # Overlay text on the frame
        overlay_text_on_frame(frame, names_dict[np.argmax(probs)], max(probs))

        # Display the frame in a window
        cv2.imshow('Frame', frame)

        # Write the frame to the output video file
        out.write(frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'messy.MP4'
    model_path = 'best.pt'
    output_path = 'output_video_espresso.mp4'

    # Define headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtdWhhbW1hZG9zYW1hLmhxQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwNjU5NzA5NCwiZXhwIjoxNzA2ODk3MDk0fQ.b8Fnd3T5Egmd_r5vyi-7u1eF175JJAjsYD8uy_1-f00",  # Replace with your actual authorization token
        "X-SERVER-TO-SERVER": "true"
    }

    main(video_path, model_path, output_path)
