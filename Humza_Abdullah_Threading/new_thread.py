import cv2
import threading
from shared_constants import frame_queue_final, frame_queue_branch,display_queue_atm, display_queue_branch
from ATM.atm_thread import process_frame as process_frame_atm
from Branch.branch_thread import process_frame_branch
import time


username = 'usama.xloop'
password = 'Xloop@123'
ip_address = '192.168.6.19'
video_url = f"rtsp://{username}:{password}@{ip_address}:554/cam/realmonitor?channel=1&subtype=1"

# Specify your video file path
video_path = 'all.mp4'

# def capture_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames_per_second = 25
#     delay_seconds = 1

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_queue_final.put(frame)
#         frame_queue_branch.put(frame)
        
#         # Add a delay to capture one frame per second
#         time.sleep(1 / frames_per_second * delay_seconds)  
        

        
#     cap.release()
    
# # Start capturing frames in separate threads
# frame_thread = threading.Thread(target=capture_frames, args=(video_path,), daemon=True)
# frame_thread.start()


# while True:
#     frame = frame_queue_final.get()
#     branch_frame = frame_queue_branch.get()

#     # Check if the capture thread has finished (if None is put in the queue)
#     if frame is None or branch_frame is None:
#         break  
            
    # # Start separate threads for processing frames
    # atm_processing_thread = threading.Thread(target=process_frame_atm, args=(frame,))
    # branch_processing_thread = threading.Thread(target=process_frame_branch, args=(branch_frame,))

    # # Start both threads
    # atm_processing_thread.start()
    # branch_processing_thread.start()

    # # Wait for both threads to finish
    # atm_processing_thread.join()
    # branch_processing_thread.join()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Signal the processing threads to stop
# frame_queue_final.put(None)
# frame_queue_branch.put(None)

# # Wait for the capture threads to finish
# frame_thread.join()

# cv2.destroyAllWindows()


# _______________________________________--

# def capture_frames(video_path):
#     cap = cv2.VideoCapture(video_path)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_queue_final.put(frame)
#         frame_queue_branch.put(frame)
        

        
#     cap.release()
    

# # Start capturing frames in a separate thread
# frame_thread = threading.Thread(target=capture_frames, args=(video_path,), daemon=True)
# frame_thread.start()




# while True:
#     frame = frame_queue_final.get()
#     branch_frame = frame_queue_branch.get()

    
    
#     # Check if the capture thread has finished (if None is put in the queue)
#     if frame is None:
#         break
    
#      # Check if there's at least one value in the queue
#     if frame_queue_final.qsize() > 0 and frame_queue_branch.qsize() > 0:
        
#         # Start separate threads for processing frames
#         atm_processing_thread = threading.Thread(target=process_frame_atm, args=(frame,))
#         branch_processing_thread = threading.Thread(target=process_frame_branch, args=(branch_frame,))
        
#         # Start both threads
#         atm_processing_thread.start()
#         branch_processing_thread.start()
        
        
#         atm_processing_thread.join()
#         branch_processing_thread.join()
    
        

#     # Send the frame for processing in final_live.py
#     # process_frame_atm(frame)
#     # process_frame_branch(branch_frame)
    

    

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Signal the processing thread to stop
# frame_queue_final.put(None)
# frame_queue_branch.put(None)


# frame_thread.join()


# cv2.destroyAllWindows()


def capture_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_queue_final.put(frame)
        frame_queue_branch.put(frame)
        

    cap.release()

# Start capturing frames in a separate thread
frame_thread = threading.Thread(target=capture_frames, args=(video_path,), daemon=True)
frame_thread.start()

while True:
    # Get frames from both queues
    frame = frame_queue_final.get()
    branch_frame = frame_queue_branch.get()

    # Check if the capture thread has finished (if None is put in the queue)
    if frame is None:
        break
        
    print("Processing Frames Completed")
        
    # Check if there's at least one value in both queues
    if frame_queue_final.qsize() > 0 and frame_queue_branch.qsize() > 0:
        
        print("Starting Function Threads in Parallel")

        # Start separate threads for processing frames
        atm_processing_thread = threading.Thread(target=process_frame_atm, args=(frame,))
        branch_processing_thread = threading.Thread(target=process_frame_branch, args=(branch_frame,))

        # Start both threads
        atm_processing_thread.start()
        branch_processing_thread.start()

        # Wait for both threads to finish
        atm_processing_thread.join()
        branch_processing_thread.join()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Signal the processing threads to stop
frame_queue_final.put(None)
frame_queue_branch.put(None)


# Wait for the capture thread to finish
frame_thread.join()

cv2.destroyAllWindows()
