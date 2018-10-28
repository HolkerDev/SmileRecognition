import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')


def detection(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (204, 14, 83), 5)
        img_gray = gray[y:y + h, x:x + w]
        img_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.5,
            minNeighbors=7,
            minSize=(24,24)

        )
        for (ix, iy, iw, ih) in eyes:
            cv2.rectangle(img_color, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 5)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smiles = smile.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=35,
            minSize=(24, 24)
        )

        for (x, y, w, h) in smiles:
            cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # return frame


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection(gray, frame)
    cv2.imshow('Face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
