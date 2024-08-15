import dlib
import cv2

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Open a connection to the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ensure the image type is correct for dlib
    # Convert gray to 8-bit grayscale
    if gray.dtype != 'uint8':
        gray = gray.astype('uint8')

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Draw a rectangle around the faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
