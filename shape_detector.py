import cv2
import numpy as np
import streamlit as st

st.title("Geometric Shape Detection with OpenCV")
st.write("Upload an image with shapes to detect and classify them!")

# Option to upload an image
uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
  st.info("👆 Please upload an image to start shape detection.")
  st.stop()

# Read the uploaded image safely
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)

if img is None:
  st.error("Could not decode the uploaded image. Please try a different file.")
  st.stop()

# Process the image for shape detection
output_img = img.copy()
gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
  approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
  if cv2.contourArea(cnt) < 100:
    continue

  cv2.drawContours(output_img, [approx], 0, (0, 0, 255), 3)
  x, y = approx.ravel()[0], approx.ravel()[1] - 10
  vertices = len(approx)

  # Classify shapes accurately based on corners and proportions
  if vertices == 3:
    shape_name = "Triangle"
  elif vertices == 4:
    # Check aspect ratio to distinguish Square from Rectangle
    _, _, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    if 0.95 <= aspect_ratio <= 1.05:
      shape_name = "Square"
    else:
      shape_name = "Rectangle"
  elif vertices == 5:
    shape_name = "Pentagon"
  elif vertices == 6:
    shape_name = "Hexagon"
  else:
    shape_name = "Circle"

  cv2.putText(
      output_img,
      shape_name,
      (x, y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 150, 0),
      2,
  )

# Convert OpenCV image (BGR) to RGB for Streamlit display
output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

st.image(
    output_img_rgb,
    caption="Processed Image with Shapes Detected",
    use_container_width=True,
)
