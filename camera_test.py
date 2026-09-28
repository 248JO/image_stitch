import cv2
import os

# create folder for captured images
OUTPUT_FOLDER = "images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# connect to the camera
camera = cv2.VideoCapture(0)

# check that the camera opened correctly
if not camera.isOpened():
    raise RuntimeError("Could not connect to camera.")

print("Camera connected.")
print("Press the space bar to take a photo.")
print("Press Q to quit.")

while True:

    # get one frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("Could not read frame.")
        break

    # show the live camera feed
    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    # save an image when space is pressed
    if key == 32:

        output_path = os.path.join(
            OUTPUT_FOLDER,
            "camera_test.jpg"
        )

        cv2.imwrite(output_path, frame)

        print("Photo saved:", output_path)

    # quit when q is pressed
    elif key == ord("q"):
        break

# disconnect the camera and close the preview window
camera.release()
cv2.destroyAllWindows()