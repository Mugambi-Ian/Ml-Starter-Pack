import cv2
import time


faceCascade = cv2.CascadeClassifier("include/xml/face.xml")
eyeCascade = cv2.CascadeClassifier("include/xml/eye.xml")
noseCascade = cv2.CascadeClassifier("include/xml/nose.xml")

video = cv2.VideoCapture(0)
while True:
    check, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1)
    coords = []
    for (x, y, w, h,) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.22)
        nose = noseCascade.detectMultiScale(roi_gray, 1.22)
        if len(eyes) == 2 and len(nose) == 1:
            coords = [x, y, w, h]
            face_ = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = face_[y:y+h, x:x+w]
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey),
                            (ex+ew, ey+eh), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in nose:
                cv2.rectangle(roi_color, (ex, ey),
                            (ex+ew, ey+eh), (255, 255, 0), 2)
    cv2.imshow("camera-stream", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
