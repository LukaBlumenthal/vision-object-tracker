import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()

    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Webcam", frame)
    cv2.imshow("HSV", hsv)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()