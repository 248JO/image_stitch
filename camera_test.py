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
    print(f"Move sample to ({x}, {y})")     #eventually: move_motor(x, y), wait for motion, capture frame, and image capture/queue
    print("Press SPACE in the preview window to capture.")
    print("Press Q to quit.") 

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Could not read camera frame.")
            break

        # make a copy just for the preview
        preview = frame.copy()

        # optional text overlay
        cv2.putText(
            preview,
            f"Position: ({x}, {y})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            preview,
            "SPACE = capture | Q = quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # show live camera preview
        cv2.imshow("Camera Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        # SPACE pressed
        if key == ord(" "):
            break

        # Q pressed
        if key == ord("q"):
            camera.release()
            cv2.destroyAllWindows()

            image_queue.join()
            image_queue.put(None)
            stitch_thread.join()

            print("Scan cancelled.")
            exit()

    if not ret:
        continue

    # camera has already been streaming,
    # so you don't really need to throw away 5 buffered frames anymore

    filename = f"test_x{x:03d}_y{y:03d}.jpg"
    path = os.path.join(OUTPUT_FOLDER, filename)


    cv2.imwrite(path, frame)

    print("saved:", path)

    # add the new image to the processing queue
    image_queue.put((frame.copy(), x, y, path))

    print("image added to queue")

camera.release()
cv2.destroyAllWindows()

# wait until all captured images have been processed
image_queue.join()

# tell tread that scan is finished
image_queue.put(None)

stitch_thread.join()

print("test image capture sucessful")