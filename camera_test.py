import cv2
import os
import time

# create folder for captured images
OUTPUT_FOLDER = "camera_test_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# connect to the camera
camera = cv2.VideoCapture(0)

# check that the camera opened correctly
if not camera.isOpened():
    raise RuntimeError("Could not connect to camera.")

print("Camera connected.")

#input from motor positions -- will be supplied from motor movements
positions = [(0, 0), (51, 0), (102, 0)]

for x, y in positions:
    input(f"press enter to simulate a sample move to ({x}, {y})")
    
    # give the camera time to settle and autofocus
    time.sleep(2)

    # grab a few frames so an old buffered frame is not being saved
    for _ in range(5):
        ret, frame = camera.read()

    if not ret:
        print("could not capture image")
        continue

    filename = f"test_x{x:03d}_y{y:03d}.jpg"
    path = os.path.join(OUTPUT_FOLDER, filename)

    cv2.imwrite(path, frame)

    print("saved:", path)

camera.release()

print("test image capture sucessful")