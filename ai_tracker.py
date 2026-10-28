import os
import cv2
import base64
import time
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


# Connect to Roboflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

print("Roboflow client created")


# Open webcam
camera = cv2.VideoCapture(1)

time.sleep(2)

last_check = 0
last_prediction = None

direction_x = ""
direction_y = ""


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera")
        break


    # Size and center of camera image
    frame_height, frame_width = frame.shape[:2]

    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2


    current_time = time.time()


    # Run AI every 2 seconds
    if current_time - last_check >= 2:

        success_encode, buffer = cv2.imencode(".jpg", frame)

        if success_encode:

            image_base64 = base64.b64encode(buffer).decode("utf-8")

            result = client.run_workflow(
                workspace_name="nutzer-lp-s-workspace",
                workflow_id="my-first-project-vmy-first-project-cqwpd-1-rfdetr-small-t1-logic",
                images={
                    "image": image_base64
                },
                use_cache=False
            )

            predictions = result[0]["predictions"]["predictions"]


            if predictions:

                # Take prediction with highest confidence
                best_prediction = max(
                    predictions,
                    key=lambda prediction: prediction["confidence"]
                )

                # Only accept predictions above 70%
                if best_prediction["confidence"] >= 0.70:

                    last_prediction = best_prediction

                    x = last_prediction["x"]
                    y = last_prediction["y"]


                    # Left / Right
                    if x < frame_center_x:
                        direction_x = "Left"
                    else:
                        direction_x = "Right"


                    # Up / Down
                    if y < frame_center_y:
                        direction_y = "Up"
                    else:
                        direction_y = "Down"


                    print(
                        "Confidence:",
                        round(last_prediction["confidence"], 2),
                        "| Direction:",
                        direction_x,
                        direction_y
                    )

                else:
                    last_prediction = None
                    print("No reliable object detected")


            else:
                last_prediction = None
                print("No object detected")


        last_check = current_time


    # Draw last detection
    if last_prediction is not None:

        x = last_prediction["x"]
        y = last_prediction["y"]

        width = last_prediction["width"]
        height = last_prediction["height"]

        confidence = last_prediction["confidence"]


        # Bounding box corners
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)

        x2 = int(x + width / 2)
        y2 = int(y + height / 2)


        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Draw object center
        cv2.circle(
            frame,
            (int(x), int(y)),
            5,
            (0, 0, 255),
            -1
        )


        # Text
        text = f"blue_cap {confidence:.2f} {direction_x} {direction_y}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Draw image center
    cv2.circle(
        frame,
        (frame_center_x, frame_center_y),
        6,
        (255, 0, 0),
        -1
    )


    # Show live camera
    cv2.imshow("AI Blue Cap Tracker", frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()