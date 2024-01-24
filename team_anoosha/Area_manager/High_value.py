import mtcnn
from mtcnn.mtcnn import MTCNN
import face_recognition
import pickle
import cv2
import os

def recognize_faces(frame, face_detector, data, save_directory):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face['box']
        encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
        matches = face_recognition.compare_faces(data["encodings"], encodings, tolerance=0.3)

        if any(matches):
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            recognize_faces.frame_counter += 1
            save_path = os.path.join(save_directory, f'recognized_frame_{recognize_faces.frame_counter}.jpg')
            cv2.imwrite(save_path, frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

recognize_faces.frame_counter = 0

def main():
    save_directory = '/home/xloop/face_recognition/High_value_customer/ImagesP/save'
    face_detector = MTCNN()
    data = pickle.loads(open('face_enc', 'rb').read())

    print("Streaming started")
    video_path = 'valery.mp4'
    video_capture = cv2.VideoCapture(video_path)

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/valery.avi', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        recognize_faces(frame, face_detector, data, save_directory)

        out.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
