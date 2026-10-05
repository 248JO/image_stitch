from machine import Pin
from time import sleep, sleep_us

DIR = Pin(17, Pin.OUT)
STEP = Pin(16, Pin.OUT)


def move(steps, direction):

    DIR.value(direction)

    for i in range(steps):

        STEP.value(1)
        sleep_us(1000)

        STEP.value(0)
        sleep_us(1000)


print("Forward")

move(200, 1)

sleep(2)

print("Backward")

move(200, 0)

print("Done")