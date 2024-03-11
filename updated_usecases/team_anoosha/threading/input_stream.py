import cv2


def capture_webcam():
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Return the frame
        yield frame
        # cv2.imshow('frame', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        cv2.imshow("frame", frame)

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "_main_":
# capture_webcam()
#     for frame in capture_webcam():
#         # You can do whatever you want with the frame here
#         # For example, you can pass it to a different function
#         # process_frame(frame)
#         pass
