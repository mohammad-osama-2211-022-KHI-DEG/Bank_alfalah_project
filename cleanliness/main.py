
import subprocess
import cv2
import re
import threading

def process_camera(yolo_command, cap, output_video, frame_width, frame_height, cup_threshold, person_threshold, laptop_threshold, chair_threshold, tv_threshold, camera_label, branch_id, city_name):
    try:
        yolo_process = subprocess.Popen(
            yolo_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        for line in yolo_process.stdout:
            line = line.strip()
            print(line)

            object_counts = {
                "cup": 0,
                "person": 0,
                "laptop": 0,
                "chair": 0,
                "bag": 0,
                "bottle": 0,
                "glasses": 0,
                "keyboard": 0,
                "mouse": 0,
                "pen": 0,
                "papers": 0,
                "tv": 0,
            }

            counts = re.findall(r"(\d+) (\w+)", line)
            for count, obj in counts:
                if obj in object_counts:
                    object_counts[obj] = int(count)

            ret, frame = cap.read()

            if ret:
                # Check scene cleanliness based on object counts
                cleanliness_status = "Clean"
                if (
                    object_counts["cup"] >= cup_threshold
                    or object_counts["person"] >= person_threshold
                    or object_counts["laptop"] >= laptop_threshold
                    and object_counts["chair"] >= chair_threshold
                    and object_counts["tv"] >= tv_threshold
                ):
                    cleanliness_status = "Messy"
                    print(f"Cleanliness Status ({camera_label}, City: {city_name}): {cleanliness_status}")

                # Add object counts and cleanliness status text to the frame
                overlay_text = f"Room ({camera_label}): {cleanliness_status}, City: {city_name}"
                cv2.putText(frame, overlay_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                output_video.write(frame)
                cv2.imshow(f"Cleanliness Status ({camera_label})", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        yolo_process.terminate()
        yolo_process.wait()

    except subprocess.CalledProcessError as e:
        print(f"Error running YOLO command: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define the threshold values for determining scene cleanliness
cup_threshold = 1
person_threshold = 2
laptop_threshold = 1
chair_threshold = 1
bag_threshold = 1
bottle_threshold = 1
glasses_threshold = 1
keyboard_threshold = 1
mouse_threshold = 1
pen_threshold = 1
papers_threshold = 1
tv_threshold = 1

# Define the YOLO command and other parameters
yolo_command = "yolo task=detect mode=predict model=yolov8m.pt show=True conf=0.5 source="

# Create VideoCapture objects for each camera
cap1 = cv2.VideoCapture("cctv.mp4")
cap2 = cv2.VideoCapture("office.mov")

# Get the original video frame dimensions
frame_width = int(cap1.get(3))
frame_height = int(cap1.get(4))

# Define the VideoWriter objects for each camera
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video1 = cv2.VideoWriter('output_video_camera1.avi', fourcc, 20.0, (frame_width // 2, frame_height // 2))
output_video2 = cv2.VideoWriter('output_video_camera2.avi', fourcc, 20.0, (frame_width // 2, frame_height // 2))

# Create threads for each camera
thread1 = threading.Thread(target=process_camera, args=(yolo_command + "cctv.mp4", cap1, output_video1, frame_width, frame_height, cup_threshold, person_threshold, laptop_threshold, chair_threshold, tv_threshold, "Karachi_Branch", 1, "Karachi"))
thread2 = threading.Thread(target=process_camera, args=(yolo_command + "office.mov", cap2, output_video2, frame_width, frame_height, cup_threshold, person_threshold, laptop_threshold, chair_threshold, tv_threshold, "Islamabad_Branch", 2, "Islamabad"))

# Start the threads
thread1.start()
thread2.start()

# Wait for threads to finish
thread1.join()
thread2.join()

# Release the VideoCapture objects, close the OpenCV windows, and release the VideoWriter objects
cap1.release()
cap2.release()
output_video1.release()
output_video2.release()
cv2.destroyAllWindows()