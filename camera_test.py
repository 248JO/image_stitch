import cv2
import os
import time
import queue
import threading
from stitch import process_image

# create folder for captured images
OUTPUT_FOLDER = "camera_test_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Queue for holding images
image_queue = queue.Queue()

#function to send stitched images to stitch.py
def stitching_worker():
    while True:

        item = image_queue.get()

        # stop the worker when there are no more images
        if item is None:
            image_queue.task_done()
            break

        frame, x, y, path = item

        process_image(
            frame,
            x,
            y,
            path
        )

        image_queue.task_done()

# start stitching thread
stitch_thread = threading.Thread(
    target=stitching_worker
)
stitch_thread.start()

# connect to the camera
camera = cv2.VideoCapture(0)

# check that the camera opened correctly
if not camera.isOpened():
    raise RuntimeError("Could not connect to camera.")

print("Camera connected.")

#input from motor positions -- will be supplied from motor movements
positions = [(0, 0), (51, 0)]

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

    # add the new image to the processing queue
    image_queue.put((frame.copy(), x, y, path))

    print("image added to queue")

camera.release()

# wait until all captured images have been processed
image_queue.join()

# tell tread that scan is finished
image_queue.put(None)

stitch_thread.join()

print("test image capture sucessful")