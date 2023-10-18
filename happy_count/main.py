from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from collections import defaultdict

# Load the pre-trained MTCNN model for face detection
detector = MTCNN()

classifier = load_model('./Emotion_Detection.h5')
# class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

class_labels = ['not Happy', 'Happy', 'not Happy', 'not Happy', 'not Happy']

# Set the desired window size
window_width = 1200
window_height = 600


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./NVR_ch8_main_20230920150003_20230920160003.mp4")

# Initialize a dictionary to store previous emotions for each unique face
previous_emotions = {}

# Initialize a dictionary to store happy count for each unique face
person_happy_count = {}

# Initialize the total happy count
happy_count = 0

# Initialize a counter for face IDs
face_id_counter = 0

# Initialize a dictionary to store the recent positions of each face
face_positions = defaultdict(list)

# Set the maximum distance threshold to associate faces with the same person
max_distance_threshold = 100

while True:
    ret, frame = cap.read()

    if not ret:  # Check if there are no more frames
        break

    labels = []

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Assign a unique ID to the detected face and display it below the face
            assigned_id = None
            for face_id, positions in face_positions.items():
                if any(np.linalg.norm(np.array([x, y]) - pos) < max_distance_threshold for pos in positions):
                    assigned_id = face_id
                    break

            if assigned_id is None:
                # If no existing ID is assigned, create a new one
                assigned_id = f'Face {face_id_counter}'
                face_id_counter += 1

            face_positions[assigned_id].append((x, y))  # Store the face position

            cv2.putText(frame, assigned_id, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the assigned_id is in the previous_emotions dictionary
            if assigned_id not in previous_emotions:
                previous_emotions[assigned_id] = label
                person_happy_count[assigned_id] = 0

            # Check if the emotion changes from something other than 'Happy' to 'Happy'
            if previous_emotions[assigned_id] != 'Happy' and label == 'Happy':
                person_happy_count[assigned_id] += 1
                happy_count += 1
                # Reset the count to 0 after incrementing
                person_happy_count[assigned_id] = 0

            # Update the previous emotion for this person
            previous_emotions[assigned_id] = label

    # Resize the frame to the desired window size
    frame = cv2.resize(frame, (window_width, window_height))

    # Display the resized frame
    cv2.putText(frame, f"Happy Count: {happy_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
