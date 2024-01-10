import subprocess
import cv2
import re

# Define the path to your custom-trained YOLO weights file
custom_weights_path = "table4.pt"

# Define the standard table size
standard_table_width = 10
standard_table_height = 10

# Create an OpenCV VideoCapture object to read the video source
cap = cv2.VideoCapture("first_part.mp4")

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit(1)

# Get the original video frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the VideoWriter object to save the output video with original frame dimensions
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))

try:
    yolo_command = f"yolo task=detect mode=predict model={custom_weights_path} show=True conf=0.5 source=first_part.mp4"
    yolo_process = subprocess.Popen(
        yolo_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Initialize variables for table size and object counter
    table_width = 0
    table_height = 0

    # Process YOLO command output line by line
    for line in yolo_process.stdout:
        line = line.strip()
        print(line)  # Print the log message to the terminal

        # Use regular expression to extract the number of objects
        match = re.search(r"(\d+) objects", line)
        if match:
            num_objects = int(match.group(1))
            #print("Number of Objects:", num_objects)

            # Extract table size (a simplified example, adjust based on your implementation)
            object_counts = {}
            counts = re.findall(r"(\d+) (\w+)", line)
            for count, obj in counts:
                object_counts[obj] = int(count)

            if 'table' in object_counts and object_counts['table'] == 1:
                table_width = object_counts.get('width', 0)
                table_height = object_counts.get('height', 0)

            # Extract object coordinates
            obj_x = object_counts.get('x', 0)
            obj_y = object_counts.get('y', 0)

            # Adjust expected_objects based on table size and location
            if (
                table_width * table_height == standard_table_width * standard_table_height
                and 0 < obj_x < standard_table_width
                and 0 < obj_y < standard_table_height
            ):
                expected_objects = 221  # Object count for standard size table
            elif (
                table_width * table_height > standard_table_width * standard_table_height
                and 0 < obj_x < table_width
                and 0 < obj_y < table_height
            ):
                expected_objects = 20  # Object count for larger table
            else:
                expected_objects = 15  # Object count for smaller table


            # Read a frame from the video
            ret, frame = cap.read()
            if ret:
                # Update conditions based on table size and object location

                if num_objects > expected_objects:
                    overlay_text = "Table Messy"
                    print("Room is Messy, Objects are: ", num_objects)
                else:
                    overlay_text = "Table Clean"
                    print("Room is Clean Objects are: ", num_objects)

                # Add object counts to the frame
                cv2.putText(frame, overlay_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Write the frame to the output video
                output_video.write(frame)

                # Show the frame with counts
                cv2.imshow("Object Counts", frame)

            # Wait for a key press and break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except subprocess.CalledProcessError as e:
    print(f"Error running YOLO command: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the YOLO process
    if 'yolo_process' in locals():
        yolo_process.terminate()
        yolo_process.wait()

    # Release the video capture, close the OpenCV windows, and release the output video
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
