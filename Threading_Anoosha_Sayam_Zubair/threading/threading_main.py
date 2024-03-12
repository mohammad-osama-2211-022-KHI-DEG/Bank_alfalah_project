import threading

import cv2
from branch_manager_LIVE import process_face as branch_manager_script
from cwt_LIVE import process_faces as cwt_script
from emotions import process_frame as emotions_script
from mtcnn import MTCNN
from queue_frame import branch_manager_frame, cwt_frame, emotion_frame

# username = 'usama.xloop'
# password = 'Xloop@123'
# ip_address = '192.168.6.19'
# video_url = f"rtsp://{username}:{password}@{ip_address}:554/cam/realmonitor?channel=1&subtype=1"

# Specify your video file path
video_path = "videos/zubair.mp4"
face_detector = MTCNN()


def capture_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        emotion_frame.put(frame)
        cwt_frame.put(frame)
        branch_manager_frame.put(frame)
        print("Frames are saved in Queue")

    cap.release()


# Start capturing frames in a separate thread
frame_thread = threading.Thread(target=capture_frames, args=(video_path,), daemon=True)
frame_thread.start()

while True:
    # Get frames from queues
    happy_angry_frame = emotion_frame.get()
    customer_frame = cwt_frame.get()
    branch_frame = branch_manager_frame.get()

    # Check if the capture thread has finished (if None is put in the queue)
    if customer_frame is None or happy_angry_frame is None or branch_frame is None:
        break

    print("Processing Frames Completed")

    print("Starting Function Threads in Parallel")
    font = cv2.FONT_HERSHEY_DUPLEX
    faces = face_detector.detect_faces(customer_frame)
    rgb = cv2.cvtColor(customer_frame, cv2.COLOR_BGR2RGB)

    # Start separate threads for processing frames
    thread_1 = threading.Thread(target=emotions_script, args=(happy_angry_frame,))
    thread_2 = threading.Thread(
        target=cwt_script,
        args=(
            faces,
            customer_frame,
            rgb,
        ),
    )
    thread_3 = threading.Thread(
        target=branch_manager_script,
        args=(
            faces,
            rgb,
            branch_frame,
            font,
        ),
    )
    # Start both threads
    thread_1.start()
    thread_2.start()
    thread_3.start()

    # Wait for threads to finish
    thread_1.join()
    thread_2.join()
    thread_3.join()

    cv2.imshow("Frame", branch_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Signal the processing threads to stop
cwt_frame.put(None)
emotion_frame.put(None)
branch_manager_frame.put(None)


# Wait for the capture thread to finish
frame_thread.join()

cv2.destroyAllWindows()