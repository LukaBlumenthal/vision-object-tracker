import cv2
import numpy as np
import time

cap = cv2.VideoCapture(1)

last_print_time = time.time()

while True:
    success, frame = cap.read()

    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        biggest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest)

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"({center_x}, {center_y})",
            (center_x + 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        frame_height, frame_width = frame.shape[:2]

        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        difference_x = center_x - frame_center_x
        difference_y = center_y - frame_center_y

        current_time = time.time()

        if current_time - last_print_time >= 2:

            if center_x < frame_center_x:
                print("Left")
            elif frame_center_x < center_x:
                print("Right")
            else:
                print("Center")
 
            if center_y < frame_center_y:
                print("Up")
            elif center_y > frame_center_y:
                print("Down")
            else:
                print("Center")

            last_print_time = current_time
             
        


    cv2.imshow("Webcam", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()