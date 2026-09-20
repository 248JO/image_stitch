import cv2
import glob
import os

files = sorted(glob.glob("images/*.jpg"))
images = []

for file in files:
    img = cv2.imread(file)
    if img is not None:
        images.append(img)

stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
status, mosaic = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/mosaic.jpg", mosaic)
    print("Success")
else:
    print("Stitch failed:", status)