# Vision Object Tracker

A small computer vision project that tracks a blue bottle cap using two different approaches:

1. **HSV color tracking** with OpenCV
2. **AI object detection** with a custom trained RF-DETR model

Both versions use a webcam, detect the object position and determine whether the object is:

- Left
- Right
- Up
- Down

The goal of this project was to compare a simple classical computer vision approach with a trained AI model.

---

## Project Structure

```text
vision-object-tracker/
├── ai_tracker.py
├── hsv_tracker.py
├── requirements.txt
├── .gitignore
├── README.md
└── images/
    ├── ai_tracker_demo.png
    ├── ai_tracker_terminal.png
    └── hsv_tracker_demo.png
```

---

# HSV Tracker

The HSV tracker uses traditional computer vision.

It does not use a trained AI model.

The webcam image is converted from BGR to HSV:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

A blue color range is then used to create a mask.

The program searches for blue areas inside the image and selects the largest detected contour.

From this contour, the tracker calculates:

- Bounding box
- Object center X
- Object center Y

The object center is then compared with the center of the camera image.

Example:

```text
Object X < Camera Center X
→ Left

Object X > Camera Center X
→ Right
```

The same logic is used for Up and Down.

## HSV Tracker Demo

![HSV Tracker Demo](images/hsv_tracker_demo.png)

## Advantages

- Simple
- Fast
- Runs locally
- No training required
- Easy to understand
- Good for simple color tracking

## Disadvantages

The HSV tracker mainly detects a **color**, not the actual object.

For example, another blue object could also be detected.

Lighting conditions can also change the HSV values and reduce detection quality.

---

# AI Tracker

The AI tracker uses a custom trained object detection model instead of only searching for a color.

The model was trained to recognize:

```text
blue_cap
```

This means the model tries to recognize the actual blue bottle cap instead of only detecting blue pixels.

---

# Training the AI Model

I created a dataset with around **75 images** of a bottle with a blue cap.

The images were uploaded and labeled in Roboflow.

The dataset was split into:

```text
53 Training Images
15 Validation Images
7 Test Images
```

A custom **RF-DETR Small** object detection model was trained using Roboflow.

The model reached approximately:

```text
mAP@50: ~97%
Precision: ~93%
Recall: 100%
```

Because the dataset is still quite small, these results should only be considered an initial test.

A larger and more varied dataset would improve the model further.

---

# How the AI Tracker Works

The webcam displays a live video.

Every two seconds, one camera frame is sent to the trained Roboflow model.

The model returns information such as:

```text
class
confidence
x
y
width
height
```

Example:

```text
Class: blue_cap
Confidence: 0.94
X: 904
Y: 428
```

Only detections with at least **70% confidence** are accepted.

```python
if best_prediction["confidence"] >= 0.70:
```

This helps reduce false detections.

The model returns the center position of the detected bottle cap.

The program compares this position with the center of the camera image.

The result can be:

```text
Left Up
Left Down
Right Up
Right Down
```

The live camera window also displays:

- Bounding box
- Object center
- Camera center
- Confidence
- Direction

---

## AI Tracker Demo

![AI Tracker Demo](images/ai_tracker_demo.png)

## Terminal Output

The program also prints the confidence and direction in the terminal.

![AI Tracker Terminal](images/ai_tracker_terminal.png)

---

# HSV vs AI

| HSV Tracker | AI Tracker |
|---|---|
| Detects color | Detects a trained object |
| Simple | More complex |
| No training required | Dataset and training required |
| Runs locally | Uses AI inference |
| Very fast | Slower |
| Sensitive to lighting | More flexible |
| Can detect other blue objects | Can learn specific objects |
| Difficult to expand | Can be extended with more classes |

The HSV version is much easier to build.

It works well when the goal is simply:

```text
Find something blue.
```

The AI version requires more work because images have to be collected, labeled and used to train a model.

However, the AI approach is much more flexible.

Instead of only detecting:

```text
blue pixels
```

the AI model can learn:

```text
blue bottle cap
```

This means the system can learn the difference between an actual object and other objects with a similar color.

---

# Why AI Is More Flexible

The AI model can be improved and expanded much more easily.

For example, the dataset could contain multiple classes:

```text
blue_cap
red_cap
green_cap
bottle
can
box
```

The model could then learn to detect different objects and different colors.

With HSV tracking, a new color usually requires a new color range.

With AI, the model can learn much more complex visual features than only color.

This makes AI more useful for future robotics applications.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/LukaBlumenthal/vision-object-tracker.git
```

Move into the project folder:

```bash
cd vision-object-tracker
```

---

## 2. Install the requirements

```bash
pip install -r requirements.txt
```

Main libraries used:

```text
OpenCV
NumPy
Roboflow Inference SDK
```

---

# Run the HSV Tracker

Start the HSV tracker with:

```bash
python3 hsv_tracker.py
```

The webcam window will open and start detecting the blue object.

Press:

```text
Q
```

to stop the program.

---

# Run the AI Tracker

The AI tracker requires a Roboflow API key.

Set the API key as an environment variable before starting the program.

On macOS or Linux:

```bash
export ROBOFLOW_API_KEY="YOUR_API_KEY"
```

The API key is not stored directly inside the Python code.

The program accesses it with:

```python
os.environ["ROBOFLOW_API_KEY"]
```

This prevents the private API key from being uploaded to GitHub.

Then start the AI tracker:

```bash
python3 ai_tracker.py
```

Press:

```text
Q
```

to stop the program.

---

# What I Learned

This project helped me understand the basic computer vision pipeline:

```text
Camera
↓
Image
↓
Object Detection
↓
Object Position
↓
Direction
```

I also learned the difference between classical computer vision and AI-based object detection.

The HSV tracker showed how an object can be detected using image processing and color ranges.

The AI tracker introduced:

- Creating a dataset
- Labeling images
- Training an object detection model
- Confidence scores
- Bounding boxes
- AI inference
- Processing webcam images
- Comparing object coordinates with the camera center

---

# Future Improvements

Possible future improvements include:

- Larger training dataset
- More object classes
- Local AI inference instead of cloud inference
- Faster detection
- Center dead zone
- Object tracking between detections
- ROS 2 integration
- Connection to a physical robot arm
- Using object coordinates to control robot movement

A future version could use the camera to detect an object and automatically move a robot arm toward it.