import cv2
import numpy as np
import os

IMG1_PATH = "images/2_x000_y000.jpg"
IMG2_PATH = "images/2_x051_y000.jpg"
OUTPUT_PATH = "outputs/2_stitched.jpg"

# make the output folder if it does not already exist
os.makedirs("outputs", exist_ok=True)

# load both images
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)

# make sure both images were loaded correctly
if img1 is None or img2 is None:
    raise RuntimeError("Could not load one or both images.")

# convert both images to grayscale so SIFT can detect features
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# create the SIFT feature detector
sift = cv2.SIFT_create()

# find keypoints and feature descriptors in each image
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

# create a brute force matcher for the SIFT descriptors
matcher = cv2.BFMatcher(cv2.NORM_L2)

# find the two closest matches for each feature
matches = matcher.knnMatch(des1, des2, k=2)

# filter out weaker matches using the Lowe ratio test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))

# stop if there are not enough reliable matches between the images
if len(good_matches) < 10:
    raise RuntimeError("Not enough reliable matches.")

# get the matching feature coordinates from both images
pts1 = np.float32(
    [kp1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

pts2 = np.float32(
    [kp2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# estimate how image 2 needs to be transformed to line up with image 1
# RANSAC helps ignore matches that do not fit the overall transformation
H, mask = cv2.findHomography(
    pts2,
    pts1,
    cv2.RANSAC,
    5.0
)

# make sure a valid homography was found
if H is None:
    raise RuntimeError("Could not estimate homography.")

print("RANSAC inliers:", int(mask.sum()))
print("Homography:")
print(H)

# get the height and width of both images
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# define the four corners of image 1
corners1 = np.float32([
    [0, 0],
    [w1, 0],
    [w1, h1],
    [0, h1]
]).reshape(-1, 1, 2)

# define the four corners of image 2
corners2 = np.float32([
    [0, 0],
    [w2, 0],
    [w2, h2],
    [0, h2]
]).reshape(-1, 1, 2)

# transform image 2's corners so we know where it will land
warped_corners2 = cv2.perspectiveTransform(corners2, H)

# combine the corners from both images
all_corners = np.concatenate(
    (corners1, warped_corners2),
    axis=0
)

# find the outer boundaries needed for the final stitched image
xmin, ymin = np.floor(
    all_corners.min(axis=0).ravel()
).astype(int)

xmax, ymax = np.ceil(
    all_corners.max(axis=0).ravel()
).astype(int)

# shift everything so none of the final coordinates are negative
translation = np.array([
    [1, 0, -xmin],
    [0, 1, -ymin],
    [0, 0, 1]
], dtype=np.float64)

# calculate the size of the final image canvas
canvas_width = xmax - xmin
canvas_height = ymax - ymin

# warp image 2 so it lines up with image 1
warped2 = cv2.warpPerspective(
    img2,
    translation @ H,
    (canvas_width, canvas_height)
)

# create a blank canvas for image 1
canvas1 = np.zeros_like(warped2)

# calculate where image 1 should be placed on the final canvas
x_offset = -xmin
y_offset = -ymin

# place image 1 onto the canvas
canvas1[
    y_offset:y_offset + h1,
    x_offset:x_offset + w1
] = img1

# create a mask showing where image 1 exists on the canvas
mask1 = np.zeros(
    (canvas_height, canvas_width),
    dtype=np.uint8
)

mask1[
    y_offset:y_offset + h1,
    x_offset:x_offset + w1
] = 255

# create a matching mask for the warped second image
mask2 = cv2.warpPerspective(
    np.ones((h2, w2), dtype=np.uint8) * 255,
    translation @ H,
    (canvas_width, canvas_height)
)

# find the areas that belong to only one image or to both images
only1 = (mask1 > 0) & (mask2 == 0)
only2 = (mask2 > 0) & (mask1 == 0)
overlap = (mask1 > 0) & (mask2 > 0)

# create a blank final image
result = np.zeros_like(warped2)

# copy over the parts that only belong to one image
result[only1] = canvas1[only1]
result[only2] = warped2[only2]

# average the pixels where the two images overlap
result[overlap] = (
    (
        canvas1[overlap].astype(np.float32)
        + warped2[overlap].astype(np.float32)
    ) / 2
).astype(np.uint8)

# save the finished stitched image
cv2.imwrite(OUTPUT_PATH, result)

print("Stitch complete.")
print("Saved:", OUTPUT_PATH)