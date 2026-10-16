import cv2
import numpy as np
import streamlit as st

st.title("Geometric Shape Detection with OpenCV")
st.write(
    "This app detects triangles, rectangles, and circles using computer"
    " vision."
)

# Option to upload an image
uploaded_file = st.file_uploader(
    "Choose an image with shapes...", type=["png", "jpg", "jpeg"]
)

img = None
if uploaded_file is not None:
  # Read uploaded image safely
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, 1)
  if img is None:
    st.error(
        "Could not decode the uploaded image. Please try a different file."
    )

# Fallback: If no file is uploaded or decoding failed, use the default shape image
if img is None:
  img = np.ones((400, 400, 3), dtype=np.uint8) * 255
  # Triangle
  pts_triangle = np.array([[200, 50], [80, 250], [320, 250]], np.int32)
  cv2.fillPoly(img, [pts_triangle], (0, 255, 0))
  # Rectangle
  cv2.rectangle(img, (50, 280), (180, 360), (255, 0, 0), -1)
  # Circle
  cv2.circle(img, (300, 320), 40, (0, 0, 255), -1)
  st.info(
      "Showing a default sample image. Upload your own image above if you like!"
  )

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

  if len(approx) == 3:
    cv2.putText(
        output_img,
        "Triangle",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 150, 0),
        2,
    )
  elif len(approx) == 4:
    cv2.putText(
        output_img,
        "Rectangle",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 150, 0),
        2,
    )
  elif len(approx) > 4:
    cv2.putText(
        output_img,
        "Circle",
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
