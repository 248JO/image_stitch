
from machine import Pin
from time import sleep, sleep_us

DIR = Pin(17, Pin.OUT)
STEP = Pin(16, Pin.OUT)

# Half-step microstepping
FULL_STEPS_PER_REV = 200
MICROSTEPS = 3
STEPS_PER_REV = FULL_STEPS_PER_REV * MICROSTEPS

TARGET_RPM = 10

# Timing at target RPM
STEP_DELAY_US = int(
    60_000_000 / (STEPS_PER_REV * TARGET_RPM)
)

# Acceleration settings
START_DELAY_US = STEP_DELAY_US * 3
RAMP_STEPS = 60


def move(steps, direction):

    DIR.value(direction)
    sleep_us(10)

    for i in range(steps):

        # Gradually accelerate and decelerate
        ramp_position = min(
            i, steps - 1 - i, RAMP_STEPS
        )

        fraction = ramp_position / RAMP_STEPS

        delay = int(
            START_DELAY_US -
            (START_DELAY_US - STEP_DELAY_US) * fraction
        )

        STEP.value(1)
        sleep_us(5)

        STEP.value(0)
        sleep_us(delay - 5)


print("Target RPM:", TARGET_RPM)
print("Microstepping: 1/2 step")

print("Forward")
move(STEPS_PER_REV, 1)

sleep(3)

print("Backward")
move(STEPS_PER_REV, 0)

print("Done")
