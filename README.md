# Shape Detection with OpenCV (very simple)

Finds shapes in an image and names them: Triangle, Square, Rectangle, Pentagon, Circle.

## Run
    pip install -r requirements.txt
    python make_sample.py        # creates images/shapes.png (once)
    python shape_detector.py     # detects shapes, saves results in /output

## How it works
Grayscale -> Blur -> Threshold -> Find contours -> Count corners -> Name the shape

| Corners | Shape |
|---|---|
| 3 | Triangle |
| 4 | Square or Rectangle (check width/height) |
| 5 | Pentagon |
| many | Circle |
