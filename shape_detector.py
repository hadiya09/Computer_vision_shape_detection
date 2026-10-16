# Simple Shape Detection with OpenCV
# Idea: find the outline of each shape, count its corners, name the shape.
import cv2

# 1. Load the image
img = cv2.imread("images/shapes.png")

# 2. Convert to grayscale and blur a little
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Threshold: shapes become white, background becomes black
_, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)

# 4. Find the outline (contour) of every shape
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. For each outline, count corners and name the shape
for c in contours:
    if cv2.contourArea(c) < 500:          # ignore tiny noise
        continue

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    corners = len(approx)

    if corners == 3:
        name = "Triangle"
    elif corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        name = "Square" if 0.95 <= w / h <= 1.05 else "Rectangle"
    elif corners == 5:
        name = "Pentagon"
    else:
        name = "Circle"

    # 6. Draw the outline and write the name
    x, y, w, h = cv2.boundingRect(c)
    cv2.drawContours(img, [c], -1, (0, 0, 0), 3)
    cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    print(name, "-", corners, "corners")

# 7. Save and show the results
cv2.imwrite("output/gray.png", gray)
cv2.imwrite("output/threshold.png", thresh)
cv2.imwrite("output/detected.png", img)
cv2.imshow("Shapes", img)      # press any key to close
cv2.waitKey(0)
cv2.destroyAllWindows()
