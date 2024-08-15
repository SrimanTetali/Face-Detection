import cv2
import sys

def main(cascPath):
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    # Check if the video capture is opened successfully
    if not video_capture.isOpened():
        print("Error opening video capture.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Check if a frame is read successfully
        if not ret:
            print("Error reading frame. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if a cascade path is provided
    if len(sys.argv) < 2:
        print("Usage: python face_detect.py <cascade_path>")
        sys.exit()

    cascPath = sys.argv[1]
    main(cascPath)
