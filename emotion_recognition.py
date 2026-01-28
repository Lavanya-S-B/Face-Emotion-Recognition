# ===============================
# Face Emotion Recognition (Fixed)
# ===============================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF warnings

import cv2
import imutils
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------
# Argument Parser
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Path to video file (optional)')
parser.add_argument('-c', '--color', type=str, default='gray',
                    choices=['gray', 'rgb', 'lab'],
                    help='Color space: gray, rgb, lab')
parser.add_argument('-b', '--bins', type=int, default=16,
                    help='Histogram bins')
parser.add_argument('-w', '--width', type=int, default=0,
                    help='Resize width (0 = no resize)')
args = vars(parser.parse_args())

# -------------------------------
# Paths
# -------------------------------
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# -------------------------------
# Load Models
# -------------------------------
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry", "disgust", "scared",
            "happy", "sad", "surprised", "neutral"]

# -------------------------------
# Video Capture
# -------------------------------
camera = cv2.VideoCapture(0)
time.sleep(2)

color = args['color']
bins = args['bins']
resizeWidth = args['width']

# -------------------------------
# Histogram Plot Setup
# -------------------------------
fig, ax = plt.subplots()
ax.set_title(f'Histogram ({color.upper()})')
ax.set_xlabel('Bin')
ax.set_ylabel('Frequency')

lw = 3
alpha = 0.5

if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros(bins), 'r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros(bins), 'g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros(bins), 'b', lw=lw, alpha=alpha)
elif color == 'lab':
    lineL, = ax.plot(np.arange(bins), np.zeros(bins), 'k', lw=lw, alpha=alpha)
    lineA, = ax.plot(np.arange(bins), np.zeros(bins), 'b', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros(bins), 'y', lw=lw, alpha=alpha)
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros(bins), 'k', lw=lw)

ax.set_xlim(0, bins - 1)
ax.set_ylim(0, 1)
plt.ion()
plt.show()

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = camera.read()
    if not ret:
        break

    if resizeWidth > 0:
        frame = imutils.resize(frame, width=resizeWidth)

    numPixels = frame.shape[0] * frame.shape[1]

    # ---------------------------
    # Histogram Processing
    # ---------------------------
    if color == 'rgb':
        (b, g, r) = cv2.split(frame)
        lineR.set_ydata(cv2.calcHist([r], [0], None, [bins], [0, 256]) / numPixels)
        lineG.set_ydata(cv2.calcHist([g], [0], None, [bins], [0, 256]) / numPixels)
        lineB.set_ydata(cv2.calcHist([b], [0], None, [bins], [0, 256]) / numPixels)

    elif color == 'lab':
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        (l, a, b) = cv2.split(lab)
        lineL.set_ydata(cv2.calcHist([l], [0], None, [bins], [0, 256]) / numPixels)
        lineA.set_ydata(cv2.calcHist([a], [0], None, [bins], [0, 256]) / numPixels)
        lineB.set_ydata(cv2.calcHist([b], [0], None, [bins], [0, 256]) / numPixels)

    else:
        gray_hist = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lineGray.set_ydata(cv2.calcHist([gray_hist], [0], None, [bins], [0, 256]) / numPixels)

    fig.canvas.draw()
    fig.canvas.flush_events()

    # ---------------------------
    # Emotion Recognition
    # ---------------------------
    frame_small = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame_small.copy()

    if len(faces) > 0:
        (fX, fY, fW, fH) = faces[0]
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi, verbose=0)[0]
        label = EMOTIONS[np.argmax(preds)]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = f"{emotion}: {prob * 100:.2f}%"
            w = int(prob * 250)
            cv2.rectangle(canvas, (5, i * 30 + 5), (w, i * 30 + 30), (0, 255, 0), -1)
            cv2.putText(canvas, text, (10, i * 30 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 255, 0), 2)

    # ---------------------------
    # Display
    # ---------------------------
    cv2.imshow("Emotion Recognition", frameClone)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
camera.release()
cv2.destroyAllWindows()
