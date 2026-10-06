import cv2
import numpy as np
import os


OUTPUT_FOLDER = "outputs"

#pre made images used when stitch.py is run directly
IMG1_PATH = "images/demo_split_left_52overlap.jpg"
IMG2_PATH = "images/demo_split_right_52overlap.jpg"

# PRELOADED_OUTPUT_PATH = os.path.join(
#     OUTPUT_FOLDER,
#     "preloaded_stitched.jpg"
# )

# Where image 2 belongs relative to image 1
# right
# left
# down
# up
DIRECTION = "right"

MIN_OVERLAP = 0.15 
MAX_OVERLAP = 0.90

LOWE_RATIO = 0.70
MIN_MUTUAL_MATCHES = 8

TRANSLATION_CLUSTER_RADIUS = 8.0
MIN_TRANSLATION_INLIERS = 8
MIN_INLIER_RATIO = 0.25


# live camera state

#these remember the previous camera frame.
previous_frame = None
previous_x = None
previous_y = None

live_pair_number = 0


# camera entry point

def process_image(frame, x, y, path):
    """
    Receives an image from the camera queue
    the first frame is stored
    then every frame after that is stitched to the previous frame
    """

    global previous_frame
    global previous_x
    global previous_y
    global live_pair_number

    print("New camera image received")
    print("Position:", (x, y))
    print("Image path:", path)

    # First camera image

    # There is nothing to stitch it to yet,
    # so just save it as the reference.
    if previous_frame is None:

        previous_frame = frame.copy()
        previous_x = x
        previous_y = y

        print("First camera image stored.")
        print("Waiting for next image...")

        return


    # second image and later

    print(
        "Stitching previous imagee",
        (previous_x, previous_y),
        "to current image",
        (x, y)
    )

    result, H = stitch_pair(
        previous_frame,
        frame,
        direction=DIRECTION,
        min_overlap=MIN_OVERLAP,
        max_overlap=MAX_OVERLAP,
        output_folder=OUTPUT_FOLDER
    )


    # save live result

    live_pair_number += 1

    output_name = (
        f"live_stitch_{live_pair_number:03d}_"
        f"x{previous_x:03d}_y{previous_y:03d}_"
        f"to_"
        f"x{x:03d}_y{y:03d}.jpg"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )

    cv2.imwrite(
        output_path,
        result
    )

    print("Live stitch complete.")
    print("Saved:", output_path)

    print("Transformation:")
    print(H)


    #the current image becomes the next reference

    previous_frame = frame.copy()
    previous_x = x
    previous_y = y

def make_overlap_masks(
    gray1,
    gray2,
    direction,
    max_overlap
):
    """
    Search only the largest plausible overlap region.
    don't assume the exact overlap percentage.

    ex: if max_overlap = 0.55 and image 2 is to
    the right of image 1:

        image 1 -> search rightmost 55%
        image 2 -> search leftmost 55%
    the actual overlap is determined later.
    """

    h1, w1 = gray1.shape
    h2, w2 = gray2.shape

    mask1 = np.zeros_like(
        gray1,
        dtype=np.uint8
    )

    mask2 = np.zeros_like(
        gray2,
        dtype=np.uint8
    )

    if direction == "right":

        # Right side of image 1
        mask1[
            :,
            int(w1 * (1 - max_overlap)):
        ] = 255

        #left side of image 2
        mask2[
            :,
            :int(w2 * max_overlap)
        ] = 255

    elif direction == "left":

        # Left side of image 1
        mask1[
            :,
            :int(w1 * max_overlap)
        ] = 255

        # right side of image 2
        mask2[
            :,
            int(w2 * (1 - max_overlap)):
        ] = 255

    elif direction == "down":

        # bottom of image 1
        mask1[
            int(h1 * (1 - max_overlap)):,
            :
        ] = 255

        #top of image 2
        mask2[
            :int(h2 * max_overlap),
            :
        ] = 255

    elif direction == "up":

        # top of image 1
        mask1[
            :int(h1 * max_overlap),
            :
        ] = 255

        # bottom of image 2
        mask2[
            int(h2 * (1 - max_overlap)):,
            :
        ] = 255

    else:
        raise ValueError(
            "direction must be right, left, down, or up"
        )

    return mask1, mask2


 
# Lowe ratio test
 

def ratio_test(knn_matches, ratio):
    """
    Applying Lowe's ratio test to SIFT descriptor matches.
    """

    good = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    return good


 
# mutual sift matching
 
def find_mutual_sift_matches(
    img1,
    img2,
    direction,
    max_overlap
):
    """
    Find sift matches inside the possible overlap area.

    A match is kept only when:

    it passes Lowe's ratio test from image 1 -> image 2.
    it passes Lowe's ratio test from image 2 -> image 1.
    both directions identify the same feature pair.

    mutual matching helps reject ambiguous matches caused by
    repetitive solar panel geometry.
    """

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    mask1, mask2 = make_overlap_masks(
        gray1,
        gray2,
        direction,
        max_overlap
    )

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(
        gray1,
        mask1
    )

    kp2, des2 = sift.detectAndCompute(
        gray2,
        mask2
    )

    print(
        "Image 1 overlap keypoints:",
        len(kp1)
    )

    print(
        "Image 2 overlap keypoints:",
        len(kp2)
    )

    if des1 is None or des2 is None:
        raise RuntimeError(
            "Not enough SIFT features found "
            "inside the possible overlap region."
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    # Image 1 to Image 2
    forward_knn = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    # Image 2 to Image 1
    reverse_knn = matcher.knnMatch(
        des2,
        des1,
        k=2
    )

    forward = ratio_test(
        forward_knn,
        LOWE_RATIO
    )

    reverse = ratio_test(
        reverse_knn,
        LOWE_RATIO
    )

    # Reverse matches have:
    #
    # queryIdx to image 2
    # trainIdx to image 1
    reverse_pairs = {
        (
            m.trainIdx,
            m.queryIdx
        )
        for m in reverse
    }

    # keep only matches that agree both ways
    mutual = [
        m
        for m in forward
        if (
            m.queryIdx,
            m.trainIdx
        ) in reverse_pairs
    ]

    print(
        "Forward ratio-test matches:",
        len(forward)
    )

    print(
        "Mutual matches:",
        len(mutual)
    )

    if len(mutual) < MIN_MUTUAL_MATCHES:
        raise RuntimeError(
            "Not enough reliable mutual SIFT matches."
        )

    return kp1, kp2, mutual


 
# translation estimation
 

def estimate_translation(
    img1,
    kp1,
    kp2,
    matches,
    direction,
    min_overlap,
    max_overlap
):
    """
    estimate a translation only transformation.

    instead of assuming one exact overlap percentage w:
    
    calculate dx/dy from every feature match
    reject movements outside the allowed overlap range
    reject movement in the wrong direction
    find the largest cluster of similar translations
    use the median of that cluster as the final translation
    calculate the actual overlap from the final translation
    """

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    # Translation proposed by each match

    # Transformation required to place image 2
    # into image 1's coordinate system.
    dx = pts1[:, 0] - pts2[:, 0]
    dy = pts1[:, 1] - pts2[:, 1]

    h, w = img1.shape[:2]

    print("\n--- Raw translation information ---")

    print(
        "Median dx from all mutual matches:",
        f"{np.median(dx):.2f}"
    )

    print(
        "Median dy from all mutual matches:",
        f"{np.median(dy):.2f}"
    )


    # convert overlap range into allowed movement range

    if direction in (
        "right",
        "left"
    ):

        # Large overlap = small movement
        min_shift = w * (
            1 - max_overlap
        )

        # Small overlap = large movement
        max_shift = w * (
            1 - min_overlap
        )

        # allow a relatively small amount of unexpected
        # vertical movement
        cross_tolerance = max(
            15.0,
            0.06 * h
        )

        if direction == "right":

            direction_ok = (
                dx > 0
            )

        else:

            direction_ok = (
                dx < 0
            )

        geometry_ok = (
            direction_ok
            & (
                np.abs(dx)
                >= min_shift
            )
            & (
                np.abs(dx)
                <= max_shift
            )
            & (
                np.abs(dy)
                <= cross_tolerance
            )
        )

    else:

        min_shift = h * (
            1 - max_overlap
        )

        max_shift = h * (
            1 - min_overlap
        )

        # allow a relatively small amount of unexpected
        # horizontal movement
        cross_tolerance = max(
            15.0,
            0.06 * w
        )

        if direction == "down":

            direction_ok = (
                dy > 0
            )

        else:

            direction_ok = (
                dy < 0
            )

        geometry_ok = (
            direction_ok
            & (
                np.abs(dy)
                >= min_shift
            )
            & (
                np.abs(dy)
                <= max_shift
            )
            & (
                np.abs(dx)
                <= cross_tolerance
            )
        )


    # keep only geometrically possible matches

    geometry_indices = np.where(
        geometry_ok
    )[0]

    print(
        "Matches consistent with allowed scan motion:",
        len(geometry_indices)
    )

    print(
        "Allowed movement range:",
        f"{min_shift:.1f} to "
        f"{max_shift:.1f} pixels"
    )

    if (
        len(geometry_indices)
        < MIN_TRANSLATION_INLIERS
    ):
        raise RuntimeError(
            "Too few matches agree with the "
            "allowed scan direction/overlap range."
        )


    # find the largest translation cluster

    candidate_dx = dx[
        geometry_indices
    ]

    candidate_dy = dy[
        geometry_indices
    ]

    best_local_indices = None
    best_cluster_size = 0
    best_mean_distance = float("inf")

    for i in range(
        len(geometry_indices)
    ):

        # how far is every candidate translation
        # from candidate i?
        translation_distance = np.sqrt(
            (
                candidate_dx
                - candidate_dx[i]
            ) ** 2
            +
            (
                candidate_dy
                - candidate_dy[i]
            ) ** 2
        )

        cluster_members = np.where(
            translation_distance
            <= TRANSLATION_CLUSTER_RADIUS
        )[0]

        cluster_size = len(
            cluster_members
        )

        if cluster_size == 0:
            continue

        # use descriptor distance as a tie breaker
        original_indices = (
            geometry_indices[
                cluster_members
            ]
        )

        mean_match_distance = np.mean([
            matches[j].distance
            for j in original_indices
        ])

        if (
            cluster_size
            > best_cluster_size
        ):
            best_cluster_size = (
                cluster_size
            )

            best_local_indices = (
                cluster_members
            )

            best_mean_distance = (
                mean_match_distance
            )

        elif (
            cluster_size
            == best_cluster_size
            and mean_match_distance
            < best_mean_distance
        ):
            best_local_indices = (
                cluster_members
            )

            best_mean_distance = (
                mean_match_distance
            )


    if best_local_indices is None:
        raise RuntimeError(
            "Could not find a valid "
            "translation cluster."
        )


    # convert cluster indices back to indices
    # in the full match list
    inlier_indices = geometry_indices[
        best_local_indices
    ]


    # Quality checks

    if (
        len(inlier_indices)
        < MIN_TRANSLATION_INLIERS
    ):
        raise RuntimeError(
            "Not enough matches agree "
            "on one translation."
        )

    inlier_ratio = (
        len(inlier_indices)
        / len(geometry_indices)
    )

    if (
        inlier_ratio
        < MIN_INLIER_RATIO
    ):
        raise RuntimeError(
            "Alignment is unreliable. "
            f"Inlier ratio = "
            f"{inlier_ratio:.2f}"
        )


    # Final translation

    final_dx = float(
        np.median(
            dx[inlier_indices]
        )
    )

    final_dy = float(
        np.median(
            dy[inlier_indices]
        )
    )


    # Calculate actual measured overlap

    if direction in (
        "right",
        "left"
    ):

        measured_overlap = (1 - abs(final_dx) / w)

    else:

        measured_overlap = (
            1
            - abs(final_dy) / h
        )


    print("\n Final alignment")

    print(
        "Estimated translation:"
    )

    print(
        "dx =",
        f"{final_dx:.2f}"
    )

    print(
        "dy =",
        f"{final_dy:.2f}"
    )

    print(
        "Measured overlap:",
        f"{measured_overlap * 100:.1f}%"
    )

    print(
        "Translation inliers:",
        len(inlier_indices)
    )

    print(
        "Inlier ratio:",
        f"{inlier_ratio:.2f}"
    )

    print(
        "-----------------------\n"
    )


    # Translation only transformation matrix

    H = np.array([
        [
            1.0,
            0.0,
            final_dx
        ],
        [
            0.0,
            1.0,
            final_dy
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ], dtype=np.float64)

    return H, inlier_indices


 
# FEATHER BLENDING
 

def feather_blend(
    img1,
    img2,
    H
):
    """
    Warp image 2 using the estimated translation and
    feather-blend the overlapping region
    """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(
        -1,
        1,
        2
    )

    corners2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(
        -1,
        1,
        2
    )

    # Find where image 2 lands
    warped_corners2 = (
        cv2.perspectiveTransform(
            corners2,
            H
        )
    )

    # Combine corners from both images
    all_corners = np.concatenate(
        (
            corners1,
            warped_corners2
        ),
        axis=0
    )

    xmin, ymin = np.floor(
        all_corners
        .min(axis=0)
        .ravel()
    ).astype(int)

    xmax, ymax = np.ceil(
        all_corners
        .max(axis=0)
        .ravel()
    ).astype(int)


    # Shift entire mosaic so coordinates are never negative

    translation = np.array([
        [
            1,
            0,
            -xmin
        ],
        [
            0,
            1,
            -ymin
        ],
        [
            0,
            0,
            1
        ]
    ], dtype=np.float64)

    canvas_width = (
        xmax - xmin
    )

    canvas_height = (
        ymax - ymin
    )


    # Warp image 2

    warped2 = cv2.warpPerspective(
        img2,
        translation @ H,
        (
            canvas_width,
            canvas_height
        )
    )


    # Place image 1

    canvas1 = np.zeros_like(
        warped2
    )

    x_offset = -xmin
    y_offset = -ymin

    canvas1[
        y_offset:
        y_offset + h1,

        x_offset:
        x_offset + w1
    ] = img1

    # masks

    mask1 = np.zeros(
        (
            canvas_height,
            canvas_width
        ),
        dtype=np.uint8
    )

    mask1[
        y_offset:
        y_offset + h1,

        x_offset:
        x_offset + w1
    ] = 255


    mask2 = cv2.warpPerspective(
        np.ones(
            (
                h2,
                w2
            ),
            dtype=np.uint8
        ) * 255,

        translation @ H,

        (
            canvas_width,
            canvas_height
        )
    )


    # determine unique and overlapping regions

    only1 = (
        (mask1 > 0)
        & (mask2 == 0)
    )

    only2 = (
        (mask2 > 0)
        & (mask1 == 0)
    )

    overlap = (
        (mask1 > 0)
        & (mask2 > 0)
    )


    result = np.zeros_like(
        warped2
    )

    result[only1] = (
        canvas1[only1]
    )

    result[only2] = (
        warped2[only2]
    )


    # feather blending

    distance1 = (
        cv2.distanceTransform(
            mask1,
            cv2.DIST_L2,
            5
        )
    )

    distance2 = (
        cv2.distanceTransform(
            mask2,
            cv2.DIST_L2,
            5
        )
    )

    weight_sum = (
        distance1
        + distance2
    )

    weight_sum[
        weight_sum == 0
    ] = 1

    weight1 = (
        distance1
        / weight_sum
    )[:, :, np.newaxis]

    weight2 = (
        distance2
        / weight_sum
    )[:, :, np.newaxis]

    blended = (
        canvas1.astype(
            np.float32
        ) * weight1

        +

        warped2.astype(
            np.float32
        ) * weight2
    )

    result[
        overlap
    ] = blended[
        overlap
    ].astype(
        np.uint8
    )

    return result


 
# DEBUG MATCH IMAGES
 

def save_match_debug_images(
    img1,
    img2,
    kp1,
    kp2,
    matches,
    inlier_indices,
    output_folder
):
    """
    Save:

    All mutual matches.
    Only matches used for the final translation.
    """

    all_match_image = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            "crop_mutual.jpg"
        ),
        all_match_image
    )


    inlier_matches = [
        matches[i]
        for i in inlier_indices
    ]

    inlier_match_image = (
        cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            inlier_matches,
            None,
            flags=(
                cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        )
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            "crop_matches_inliers.jpg"
        ),
        inlier_match_image
    )


 
# COMPLETE PAIRWISE STITCH
 

def stitch_pair(
    img1,
    img2,
    direction="right",
    min_overlap=0.15,
    max_overlap=0.55,
    output_folder="outputs"
):
    """
    Complete stitching workflow
    """

    if not (
        0
        < min_overlap
        < max_overlap
        < 1
    ):
        raise ValueError(
            "Overlap range must satisfy: "
            "0 < min_overlap < max_overlap < 1"
        )


    # SIFT matching

    kp1, kp2, matches = (
        find_mutual_sift_matches(
            img1,
            img2,
            direction,
            max_overlap
        )
    )


    # translation estimation

    H, inlier_indices = (
        estimate_translation(
            img1,
            kp1,
            kp2,
            matches,
            direction,
            min_overlap,
            max_overlap
        )
    )


    # save debugging images

    save_match_debug_images(
        img1,
        img2,
        kp1,
        kp2,
        matches,
        inlier_indices,
        output_folder
    )


    # blend

    result = feather_blend(
        img1,
        img2,
        H
    )

    return result, H


 
# main test

if __name__ == "__main__":

    output_name = ("demo_stitched.jpg" )

    OUTPUT_PATH = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    img1 = cv2.imread(
        IMG1_PATH
    )

    img2 = cv2.imread(
        IMG2_PATH
    )

    if (
        img1 is None
        or img2 is None
    ):
        raise RuntimeError(
            "Could not load one or both images. "
            "Check IMG1_PATH and IMG2_PATH."
        )


    print("\nStarting stitch...")
    print("Direction:", DIRECTION)

    print(
        "Allowed overlap:",
        f"{MIN_OVERLAP * 100:.0f}% "
        "to "
        f"{MAX_OVERLAP * 100:.0f}%"
    )

    print()


    result, H = stitch_pair(
        img1,
        img2,
        direction=DIRECTION,
        min_overlap=MIN_OVERLAP,
        max_overlap=MAX_OVERLAP,
        output_folder=OUTPUT_FOLDER
    )


    cv2.imwrite(
        OUTPUT_PATH,
        result
    )


    print("Stitch complete.")
    print("Saved:", OUTPUT_PATH)

    print(
        "Final transformation:"
    )

    print(H)