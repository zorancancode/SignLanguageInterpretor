import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import Label, Button
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image, ImageTk  # Import Pillow for image handling

# Initialize camera, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Parameters for cropping and resizing
offset = 20
imgSize = 300
labels = ["A", "E", "M", "N", "S", "T"]

# Set up Tkinter window for Dark Mode
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("800x600")
root.config(bg="#2e2e2e")  # Dark background for the main window

# Add a Label for the prediction with a larger font and white text
prediction_label = Label(root, text="Prediction: ", font=("Arial", 24), fg="white", bg="#2e2e2e")
prediction_label.pack(pady=20)

# Create a canvas to display the live video
canvas = tk.Canvas(root, width=640, height=480, bg="#2e2e2e")
canvas.pack()

# Start/Stop button to control the video capture
capture_active = False


def start_capture():
    global capture_active
    capture_active = True
    start_button.config(state=tk.DISABLED, bg="#4CAF50", fg="white")  # Green button for active
    stop_button.config(state=tk.NORMAL, bg="#f44336", fg="white")  # Red button for stop
    update_frame()


def stop_capture():
    global capture_active
    capture_active = False
    start_button.config(state=tk.NORMAL, bg="#008CBA", fg="white")  # Blue button for start
    stop_button.config(state=tk.DISABLED, bg="#f44336", fg="white")  # Red button for stop


def update_frame():
    if capture_active:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera.")
            return

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure crop indices are within bounds
            hImg, wImg, _ = img.shape
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, wImg)
            y2 = min(y + h + offset, hImg)

            # Perform the crop
            imgCrop = img[y1:y2, x1:x2]

            # Ensure the crop is valid
            if imgCrop.size == 0:
                return

            # Create a white image and resize the cropped hand
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = min(math.ceil(k * w), imgSize)  # Clamp width
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = min(math.ceil(k * h), imgSize)  # Clamp height
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Perform prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Update the prediction label in the GUI
            prediction_label.config(text=f"Prediction: {labels[index]}")

            # Draw bounding box around hand with color contrast
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 255, 255), 4)  # White bounding box

        # Convert image to RGB and then to PhotoImage format for Tkinter
        img_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)  # Convert to PIL image
        img_tk = ImageTk.PhotoImage(image=img_pil)  # Convert to ImageTk object

        # Update the canvas with the new image
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk  # Keep reference to the image object

        # Recursively call update_frame every 10 ms
        canvas.after(10, update_frame)


# Add Start and Stop buttons with styled colors
start_button = Button(root, text="Start", font=("Arial", 14), command=start_capture, bg="#008CBA", fg="white")
start_button.pack(side=tk.LEFT, padx=20, pady=20)

stop_button = Button(root, text="Stop", font=("Arial", 14), command=stop_capture, state=tk.DISABLED, bg="#f44336",
                     fg="white")
stop_button.pack(side=tk.LEFT, padx=20, pady=20)

# Main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
