import cv2
from mtcnn import MTCNN
import face_recognition
import time

face_detector = MTCNN()
face_encodings_dict = {}
face_details_dict = {}
last_appearance_times = {}
last_detected_times = {}
id_threshold = 1 # generate !!ALERT!! on terminal when id disappeared

current_face_id = 0


# Function to generate unique face_id
def generate_unique_id():
    global current_face_id
    current_face_id += 1
    return str(current_face_id)

# Function to update the face encodings dictionary
def update_face_encodings(encoding, face_id):
    face_encodings_dict[face_id] = encoding

prev_elapsed = {}
# Function to check availability of face in frame
def update_face_details(face_id, start_frame, fps):
    global prev_elapsed
    if face_id not in face_details_dict:
        face_details_dict[face_id] = {'start_frame': start_frame, 'total_duration': 0}
        prev_elapsed[face_id] = 0
        last_detected_times[face_id] = current_time
    else:
        current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        elapsed_frames = current_frame - face_details_dict[face_id]['start_frame']
        print(f'face id: {face_id}')
        elapsed_time = elapsed_frames / fps
        prev_elapsed[face_id] = face_details_dict[face_id]['total_duration']  # Store current as previous for next update
        print(f'prev elapsed time: {prev_elapsed[face_id]}')
        face_details_dict[face_id]['total_duration'] += elapsed_time
        face_details_dict[face_id]['start_frame'] = current_frame
        # Update the last appearance time for the face ID
        last_appearance_times[face_id] = time.time()

# Function to check and generate alerts
def check_and_generate_alerts():
    current_time = time.time()
    for face_id, last_appearance_time in last_appearance_times.items():
        elapsed_time_since_last_appearance = current_time - last_appearance_time
        # Check if the reappearance threshold is exceeded
        if elapsed_time_since_last_appearance > id_threshold:
            total_duration = face_details_dict[face_id]['total_duration']
            # Print alert in red color
            print(f"\033[91mAlert for Face ID: {face_id}, Total Duration: {total_duration} seconds\033[0m")


video_path = 'your_video_path'
video_capture = cv2.VideoCapture(video_path)

# Get the original fps of the video
fps = video_capture.get(cv2.CAP_PROP_FPS)
print(f'frame_rate: {fps}')

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('[video_output_path].avi', fourcc, fps, (frame_width, frame_height))
while True:
    ret, frame = video_capture.read()
    # Break the loop if the video has ended
    if not ret:
        break
    font = cv2.FONT_HERSHEY_DUPLEX
    # Detect faces using MTCNN
    faces = face_detector.detect_faces(frame)
    current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
    print(f'current_frame_of_video: {current_frame}')
    current_time = current_frame/fps
    print(f'current_time_of_video: {current_time}')
    # Convert the face region to RGB (required for face_recognition)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for face in faces:
        # Extract face coordinates
        x, y, width, height = face['box']
        confidence = face['confidence']

        # Create face embeddings from current frame
        if confidence > 0.95:
            encodings = face_recognition.face_encodings(rgb, [(y, x + width, y + height, x)], num_jitters=20)
            if len(encodings) > 0:
                encoding = encodings[0]
                matches = face_recognition.compare_faces(list(face_encodings_dict.values()), encoding, tolerance=0.55)
                if any(matches):
                    # Face recognized
                    match_index = matches.index(True)
                    recognized_person_id = list(face_encodings_dict.keys())[match_index]
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 100, 255), 4)
                    cv2.putText(frame, recognized_person_id, (x + 6, y - 6), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
                    cv2.putText(frame, recognized_person_id, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, "face_recognition", (x, y + height+ 120), font, 0.5, (255, 255, 255), 1)
                    update_face_details(recognized_person_id,  video_capture.get(cv2.CAP_PROP_POS_FRAMES), fps)
                    # Display elapsed time on the video
                    elapsed_time = face_details_dict[recognized_person_id]['total_duration']
                    print(f'elapsed time: {elapsed_time}')
                    cv2.putText(frame, f"Time: {round(float(elapsed_time),2)}s", (x, y + height+10), font, 0.5, (0, 0, 0), 8, cv2.LINE_AA)
                    cv2.putText(frame, f"Time: {round(float(elapsed_time),2)}s", (x, y + height+10), font, 0.5, (255, 255, 255), 1)
                    face_id = recognized_person_id
                    time_since_last_detected = elapsed_time - prev_elapsed[face_id]
                    print(f'time difference: {time_since_last_detected}')
                    if time_since_last_detected > 1.0:
                        face_details_dict[recognized_person_id]['total_duration'] = 0
                else:
                    new_face_id = generate_unique_id()
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
                    cv2.putText(frame, new_face_id, (x + 6, y + height - 6), font, 0.5, (255, 255, 255), 1)
                    update_face_encodings(encoding, new_face_id)

    out.write(frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Check and generate alerts
    check_and_generate_alerts()

video_capture.release()
out.release()
cv2.destroyAllWindows()
