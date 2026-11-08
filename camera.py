import cv2
import os
import time
import queue
import threading
import serial
from datetime import datetime

from stitch import (
    process_image,
    finish_stitching,
    MIN_OVERLAP,
    MAX_OVERLAP
)

# SETTINGS

OUTPUT_FOLDER = "camera_test_output"

USE_MOTOR = False

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# capture numbers restart every run, so the run time
# keeps new captures from overwriting older ones
RUN_STAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

# Pico serial settings
MOTOR_PORT = "/dev/cu.usbmodem101"
BAUD_RATE = 115200

# how long to wait after motor movement
# before using the camera
CAMERA_SETTLE_SECONDS = 1.0

# remove frames that may have been captured
# while the motor was moving
CAMERA_FLUSH_FRAMES = 8


# IMAGE STITCHING QUEUE
image_queue = queue.Queue()


def stitching_worker():

    while True:

        item = image_queue.get()

        if item is None:
            image_queue.task_done()
            break

        frame, x, y, path = item

        try:

            process_image(
                frame,
                x,
                y,
                path
            )

        except Exception as error:

            print(
                "Image saved, but NOT added to the mosaic:",
                error
            )

            if not USE_MOTOR:

                print(
                    "Motor is off, so the camera is not moving. "
                    "To stitch in test mode, shift the sample by hand "
                    f"{(1 - MAX_OVERLAP) * 100:.0f}-"
                    f"{(1 - MIN_OVERLAP) * 100:.0f}% of the frame "
                    "between captures, in the direction of the "
                    "coordinates you enter."
                )

        finally:

            image_queue.task_done()


# MOTOR SERIAL FUNCTIONS

def read_motor_line():

    raw = motor.readline()

    if not raw:
        return None

    line = raw.decode(
        errors="ignore"
    ).strip()

    if line:
        print(
            "Motor:",
            line
        )

    return line


def wait_for_motor_ready():

    print(
        "Waiting for motor controller..."
    )

    # remove old serial messages
    motor.reset_input_buffer()

    # ask Pico whether setup/calibration is finished
    motor.write(
        b"STATUS\n"
    )

    motor.flush()

    while True:

        line = read_motor_line()

        if line is None:
            continue

        if line == "READY":

            print(
                "Motor controller ready."
            )

            return


def move_motor(x, y):

    command = f"{x},{y}\n"

    print()
    print("Sending coordinate:", (x, y))

    # remove old messages before sending
    # the next command
    motor.reset_input_buffer()

    motor.write(
        command.encode()
    )

    motor.flush()

    expected_message = (
        f"DONE_XY,{x},{y}"
    )

    while True:

        line = read_motor_line()

        if line is None:
            continue

        # catch motor errors
        if line.startswith("ERROR"):

            raise RuntimeError(
                f"Motor controller error: {line}"
            )

        # motor finished
        if line == expected_message:

            print(
                "Motor reached coordinate:",
                (x, y)
            )

            return


# CAMERA FUNCTIONS

def flush_camera_frames():

    print(
        "Clearing buffered camera frames..."
    )

    for _ in range(
        CAMERA_FLUSH_FRAMES
    ):

        camera.grab()


def capture_image(x, y):

    print(
        "Press SPACE to capture."
    )

    print(
        "Press Q to finish the scan."
    )

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "Could not read camera frame."
            )

            return None

        preview = frame.copy()

        cv2.putText(
            preview,
            f"Motor position: ({x}, {y})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            preview,
            "SPACE = capture | Q = finish",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Camera Preview",
            preview
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        # SPACE
        if key == ord(" "):

            return frame.copy()

        # Q
        if key == ord("q"):

            return None


# CONNECT TO MOTOR

motor = None

if USE_MOTOR:

    print("Connecting to motor controller...")

    motor = serial.Serial(
        MOTOR_PORT,
        BAUD_RATE,
        timeout=0.5
    )

    wait_for_motor_ready()

else:

    print("Motor disabled - running camera-only test.")

# CONNECT TO CAMERA
camera = cv2.VideoCapture(0)

if not camera.isOpened():

    if motor is not None:
        motor.close()

    raise RuntimeError(
        "Could not connect to camera."
    )

print(
    "Camera connected."
)


# START STITCHING THREAD

stitch_thread = threading.Thread(
    target=stitching_worker
)

stitch_thread.start()


# MAIN SCAN LOOP

capture_number = 0

try:

    while True:

        print()

        user_input = input(
            "Enter X,Y step coordinate "
            "or Q to finish: "
        ).strip()

        # finish scan
        if user_input.lower() == "q":
            break

        try:

            x_text, y_text = (
                user_input.split(",")
            )

            x = int(
                x_text.strip()
            )

            y = int(
                y_text.strip()
            )

            if x < 0 or y < 0:

                raise ValueError

        except ValueError:

            print("Invalid coordinate.")

            print(
                "Example: 500,1000"
            )

            continue


        # MOVE MOTOR
        if USE_MOTOR:
            move_motor(
                x,
                y
            )
        else:
            print(
                f"Motor disabled. Pretending position is ({x}, {y})."
            )


        # WAIT FOR CAMERA / MECHANICS TO SETTLE
        print(
            f"Waiting {CAMERA_SETTLE_SECONDS} "
            "seconds for movement to settle..."
        )

        time.sleep(
            CAMERA_SETTLE_SECONDS
        )


        # remove any frames captured while
        # the motors were moving
        flush_camera_frames()


        # CAPTURE IMAGE

        frame = capture_image(
            x,
            y
        )

        if frame is None:

            print(
                "Finishing scan."
            )

            break


        # SAVE IMAGE
        

        capture_number += 1

        filename = (
            f"capture_{RUN_STAMP}"
            f"_{capture_number:03d}"
            f"_x{x}_y{y}.jpg"
        )

        path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        saved = cv2.imwrite(
            path,
            frame
        )

        if not saved:

            print(
                "ERROR: image was not saved."
            )

            continue

        print(
            "Saved:",
            path
        )

        # SEND IMAGE TO STITCHING THREAD

        image_queue.put(
            (
                frame.copy(),
                x,
                y,
                path
            )
        )

        print(
            "Image added to stitching queue."
        )

        # with no motor move to overlap with, finish
        # stitching first so its output doesn't print
        # over the next coordinate prompt
        if not USE_MOTOR:
            image_queue.join()


finally:

    # close camera
    camera.release()

    cv2.destroyAllWindows()

    # wait for every captured image
    # to finish processing
    image_queue.join()

    # stop worker thread
    image_queue.put(None)

    stitch_thread.join()

    # make final mosaic
    finish_stitching()

    print(
        "Images captured:",
        capture_number
    )

    # close motor serial connection
    if motor is not None:
        motor.close()

    print(
        "Scan finished."
    )