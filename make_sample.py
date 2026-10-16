# Creates a practice image with 5 shapes (run once)
import cv2
import numpy as np

img = np.full((400, 700, 3), 255, np.uint8)                       # white canvas
cv2.circle(img, (100, 100), 60, (0, 0, 200), -1)                  # circle
cv2.rectangle(img, (230, 40), (350, 160), (0, 150, 0), -1)        # square
cv2.rectangle(img, (440, 60), (640, 140), (200, 80, 0), -1)       # rectangle
cv2.fillPoly(img, [np.array([[100, 230], [30, 360], [170, 360]])], (0, 140, 255))  # triangle
cv2.fillPoly(img, [np.array([[400, 220], [470, 270], [445, 350], [355, 350], [330, 270]])], (150, 0, 150))  # pentagon
cv2.imwrite("images/shapes.png", img)
