import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters for cropping and resizing
offset = 20
imgSize = 300
folder = "Data/T"
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        break

    # Detect hands
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
            continue

        # Create a white image and resize the cropped hand
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = min(math.ceil(k * w), imgSize)  # Ensure width does not exceed imgSize
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = min(math.ceil(k * h), imgSize)  # Ensure height does not exceed imgSize
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display the cropped and resized images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original camera feed
    cv2.imshow("Image", img)

    # Save the image when "s" is pressed
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image {counter} saved.")

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
