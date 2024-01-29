import subprocess
import cv2
import re
import requests
from datetime import datetime, timezone, timedelta

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit(1)
    return cap

def initialize_video_writer(output_path, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

def initialize_yolo_process(weights_path, video_source):
    yolo_command = f"yolo task=detect mode=predict model={weights_path} show=True conf=0.5 source={video_source}"
    return subprocess.Popen(
        yolo_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

def process_yolo_output(line):
    object_counts = {}
    counts = re.findall(r"(\d+) (\w+)", line)
    for count, obj in counts:
        object_counts[obj] = int(count)
    return object_counts

def calculate_level_of_mess(line):
    match = re.search(r"(\d+) Trash(?:s)?", line)
    return float(match.group(1)) / 1000 * 100 if match else 0.0

def build_payload(timestamp, level_of_mess, cleanliness_status):
    return {
        "country": "pakistan",
        "branch": "clifton",
        "city": "karachi",
        "timestamp": timestamp,
        "cleanlinessState": cleanliness_status,
        "levelOfMess": level_of_mess,
    }

def post_data_to_endpoint(endpoint_url, tag, headers, payload):
    response = requests.post(endpoint_url.format(TAG=tag), json=payload, headers=headers)
    if response.status_code == 200:
        print("Data posted successfully!")
    else:
        print(f"Failed to post data. Status code: {response.status_code}")

def main():
    custom_weights_path = "best2.pt"
    video_source_path = "alfa_messy_video.mp4"
    output_video_path = "output_video.mp4"
    endpoint_url = "http://13.235.71.140:5000/cleanliness/{TAG}"
    tag = "atm"
    jwt_token = "eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJCQUxGIiwic3ViIjoiSldUIFRva2VuIiwidXNlcm5hbWUiOiJtdWhhbW1hZG9zYW1hLmhxQGdtYWlsLmNvbSIsImF1dGhvcml0aWVzIjoiQ1JFQVRFX1VTRVIsVklFVyIsImlhdCI6MTcwNjM0MzI0OCwiZXhwIjoxNzA2NjQzMjQ4fQ.itpLlCgVvdJX5wbx76sR6Nvz87YcnFUDl5Gs0DsOCUA"

    headers = {
        "Content-Type": "application/json",
        "Authorization": jwt_token,
        "X-SERVER-TO-SERVER": "true"
    }

    cap = initialize_video_capture(video_source_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_video = initialize_video_writer(output_video_path, frame_width, frame_height)
    yolo_process = initialize_yolo_process(custom_weights_path, video_source_path)
    notification_url = "http://13.235.71.140:5000/notification"
    previous_cleanliness_status = None
    previous_level_of_mess = None
    
    try:
        for line in yolo_process.stdout:
            line = line.strip()
            print(line)

            object_counts = process_yolo_output(line)
            cleanliness_status = "MESSY" if any(count > 0 for count in object_counts.values()) else "CLEAN"

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            overlay_text = f"Object Counts: {', '.join([f'{obj}: {count}' for obj, count in object_counts.items()])}"
            level_of_mess = calculate_level_of_mess(line)

            if cleanliness_status == "CLEAN":
                level_of_mess = 0.0
                print(level_of_mess)
            else:
                cv2.putText(frame, f"Level of Mess: {level_of_mess}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print(level_of_mess)

            cv2.putText(frame, overlay_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, cleanliness_status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            timestamp = datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%dT%H:%M:%S%z')

            if cleanliness_status != previous_cleanliness_status or level_of_mess != previous_level_of_mess:
                
                payload = build_payload(timestamp, level_of_mess, cleanliness_status)
                post_data_to_endpoint(endpoint_url, tag, headers, payload)

                if level_of_mess > 0.5:
                    notification_message = f"High level of mess detected! {level_of_mess}% mess."
                    notification_payload = {
                        "timestamp": timestamp,
                        "message": notification_message,
                        "country": "pakistan",
                        "city": "karachi",
                        "branch": "clifton",
                        "usecase": "atm_cleanliness",
                    }
                    response = requests.post(notification_url, json=notification_payload)
                    print(response)
                    print(response.content)
                    if response.status_code == 200:
                        print("Notification sent successfully!")
                    else:
                        print(f"Failed to send notification. Status code: {response.status_code}")

                previous_cleanliness_status = cleanliness_status
                previous_level_of_mess = level_of_mess

            output_video.write(frame)
            cv2.imshow("Object Counts", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except subprocess.CalledProcessError as e:
        print(f"Error running YOLO command: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'yolo_process' in locals():
            yolo_process.terminate()
            yolo_process.wait()

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()