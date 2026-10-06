from machine import Pin
from time import sleep, sleep_us

DIR = Pin(17, Pin.OUT)
STEP = Pin(16, Pin.OUT)

STEPS_PER_REV = 200
TARGET_RPM = 10

# Time between steps for target RPM
STEP_DELAY_US = int(60_000_000 / (STEPS_PER_REV * TARGET_RPM))


def move(steps, direction):

    DIR.value(direction)

    for i in range(steps):

        STEP.value(1)
        sleep_us(5)

        STEP.value(0)
        sleep_us(STEP_DELAY_US - 5)


print("Target RPM:", TARGET_RPM)
print("Forward")

# One full revolution
move(200, 1)

sleep(3)

print("Backward")

# One full revolution
move(200, 0)

print("Done")