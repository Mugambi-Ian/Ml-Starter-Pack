import cv2
import time


video = cv2.VideoCapture(0)
while True:
    check, img = video.read()
    eye_cascade = cv2.CascadeClassifier("include/xml/haarcascade_eye.xml")
    face_cascade = cv2.CascadeClassifier(
        "include/xml/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1)
    for (x, y, w, h,) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.25)
        if len(eyes) > 1 & len(eyes) < 2:
            print(eyes)
            face_ = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = face_[y:y+h, x:x+w]
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)
    cv2.imshow("camera-stream", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
