import cv2

eye_cascade = cv2.CascadeClassifier("include/xml/haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(
    "include/xml/haarcascade_frontalface_default.xml")
src = cv2.imread("include/img/test.jpg")
img = cv2.resize(src, (int(src.shape[1]/2), int(src.shape[0]/2)))
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
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
