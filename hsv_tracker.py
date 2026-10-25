import cv2
import numpy as np
import time


CAMERA_INDEX = 1
PRINT_INTERVAL = 2

LOWER_BLUE = np.array([100, 100, 50])
UPPER_BLUE = np.array([140, 255, 255])


def create_blue_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        LOWER_BLUE,
        UPPER_BLUE
    )

    return mask


def find_biggest_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)

    return biggest


def get_object_center(x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2

    return center_x, center_y


def get_frame_center(frame):
    frame_height, frame_width = frame.shape[:2]

    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    return frame_center_x, frame_center_y


def get_directions(center_x, center_y, frame_center_x, frame_center_y):

    if center_x < frame_center_x:
        direction_x = "Left"
    elif center_x > frame_center_x:
        direction_x = "Right"
    else:
        direction_x = "Center"

    if center_y < frame_center_y:
        direction_y = "Up"
    elif center_y > frame_center_y:
        direction_y = "Down"
    else:
        direction_y = "Center"

    return direction_x, direction_y


def draw_tracking(frame, x, y, w, h, center_x, center_y):

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


def main():

    cap = cv2.VideoCapture(CAMERA_INDEX)

    last_print_time = time.time()

    while True:

        success, frame = cap.read()

        if not success:
            break

        mask = create_blue_mask(frame)

        biggest = find_biggest_contour(mask)

        if biggest is not None:

            x, y, w, h = cv2.boundingRect(biggest)

            center_x, center_y = get_object_center(
                x, y, w, h
            )

            frame_center_x, frame_center_y = get_frame_center(
                frame
            )

            direction_x, direction_y = get_directions(
                center_x,
                center_y,
                frame_center_x,
                frame_center_y
            )

            draw_tracking(
                frame,
                x,
                y,
                w,
                h,
                center_x,
                center_y
            )

            current_time = time.time()

            if current_time - last_print_time >= PRINT_INTERVAL:

                print(
                    f"X: {direction_x}, Y: {direction_y}"
                )

                last_print_time = current_time

        cv2.imshow("Webcam", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()