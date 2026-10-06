from machine import Pin
from time import sleep_ms

DIR = Pin(17, Pin.OUT)
STEP = Pin(16, Pin.OUT)

STEP.value(0)
DIR.value(1)

x = 0
y = 0


def move(steps, direction):
    DIR.value(direction)
    sleep_ms(10)

    for _ in range(steps):
        STEP.value(1)
        sleep_ms(5)

        STEP.value(0)
        sleep_ms(20)


print("PICO_READY")

while True:
    #wait for command from the pc
    command = input().strip()

    if command.startswith("MOVE"):
        parts = command.split()

        if len(parts) == 2:
            steps = int(parts[1])

            move(steps, 1)

            #update fake coordinate
            x += 1

            #send coordinate back to the pc
            print(f"COORD {x} {y}")
            print("DONE")