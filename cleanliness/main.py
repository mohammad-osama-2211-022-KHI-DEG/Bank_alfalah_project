
import subprocess
import cv2
import re

# Define the YOLO command
yolo_command = "yolo task=detect mode=predict model=yolov8m.pt show=True conf=0.5 source=office.mov"

# Create an OpenCV VideoCapture object to read the video source
cap = cv2.VideoCapture("office.mov")

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit(1)

# Get the original video frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

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
output_frame_width = frame_width // 2
output_frame_height = frame_height // 2

# Define the VideoWriter object to save the output video with original frame dimensions
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (output_frame_width, output_frame_height))

# Run the YOLO command and capture the output
try:
    yolo_process = subprocess.Popen(
        yolo_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Process YOLO command output line by line
    for line in yolo_process.stdout:
        line = line.strip()
        print(line)  # Print the log message to the terminal

        # Extract and update object counts
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

        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Check scene cleanliness based on object counts
            if (
                object_counts["cup"] >= cup_threshold
                or object_counts["person"] >= person_threshold
                or object_counts["laptop"] >= laptop_threshold
                and object_counts["chair"] >= chair_threshold
                and object_counts["tv"] >= tv_threshold
            ):
                cleanliness_status = "Messy"
                print(cleanliness_status)
           
            else:
                cleanliness_status = "Clean"

            # Add object counts and cleanliness status text to the frame
            overlay_text = f"Room: {cleanliness_status}"
            cv2.putText(frame, overlay_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the frame to the output video
            output_video.write(frame)

            # Show the frame with counts and status
            cv2.imshow("Cleanliness Status", frame)

        # Wait for a key press and break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close the YOLO process
    yolo_process.terminate()
    yolo_process.wait()

except subprocess.CalledProcessError as e:
    print(f"Error running YOLO command: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Release the video capture, close the OpenCV windows, and release the output video
cap.release()
output_video.release()
cv2.destroyAllWindows()
