import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()