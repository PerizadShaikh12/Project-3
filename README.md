# Driver Drowsiness Detection System

## Overview
This Python project uses OpenCV to capture images from a webcam and feeds them into a Deep Learning model to classify whether a person’s eyes are 'Open' or 'Closed'. The steps involved in the project are as follows:

1. Capture image input from a camera.
2. Detect the face in the image and create a Region of Interest (ROI).
3. Detect the eyes from ROI and feed them to the classifier.
4. Classify whether eyes are open or closed.
5. Calculate a score to check whether the person is drowsy and sound an alarm if needed.

## Dataset
The dataset comprises around 7000 images of people’s eyes labeled as ‘Open’ or ‘Closed’. The model, trained on this dataset, is saved in the `models/cnnCat2.h5` file. You can use this pre-trained model or download the dataset to train your own model.

## Model Architecture
The Convolutional Neural Network (CNN) used in this project includes:

- Convolutional Layer: 32 nodes, kernel size 3
- Convolutional Layer: 32 nodes, kernel size 3
- Convolutional Layer: 64 nodes, kernel size 3
- Fully Connected Layer: 128 nodes
- Output Layer: 2 nodes (using Softmax activation)

All layers except the output layer use the ReLU activation function.

## Prerequisites
You need to have the following installed on your system:

- Python 3.6
- OpenCV: `pip install opencv-python`
- TensorFlow: `pip install tensorflow`
- Keras: `pip install keras`
- Pygame: `pip install pygame`

Additionally, you need a webcam to capture images.

## Project Files
The project directory contains the following files and folders:

- `haar cascade files/`: XML files for object detection (face and eyes).
- `models/`: Contains the pre-trained model `cnnCat2.h5`.
- `alarm.wav`: Audio file played when drowsiness is detected.
- `Model.py`: Script used to train the CNN model.
- `Drowsiness detection.py`: Main script for the detection system.

## Steps for Detection

### Step 1: Capture Image from Camera
```python
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    ...
```
### Step 2: Detect Face and Create ROI
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
```
### Step 3: Detect Eyes from ROI and Classify
```python
left_eye = leye.detectMultiScale(gray)
right_eye = reye.detectMultiScale(gray)
for (x, y, w, h) in right_eye:
    r_eye = frame[y:y+h, x:x+w]
    ...
    rpred = model.predict_classes(r_eye)
    ...
for (x, y, w, h) in left_eye:
    l_eye = frame[y:y+h, x:x+w]
    ...
    lpred = model.predict_classes(l_eye)
    ...
```
### Step 4:  Categorize Eyes as Open or Closed
```python
if (rpred[0] == 0 and lpred[0] == 0):
    score += 1
    cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
else:
    score -= 1
    cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
```
### Step 5:  Categorize Eyes as Open or Closed
```python
if score > 15:
    try:
        sound.play()
    except:
        pass
    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
```
## Execution
To run the driver drowsiness detection system, navigate to the project directory in the command prompt and execute the following command:
```bash
python drowsiness detection.py
```
## Conclusion
This project demonstrates a practical application of computer vision and deep learning to enhance driver safety by detecting drowsiness and alerting the driver, thereby potentially preventing accidents.
