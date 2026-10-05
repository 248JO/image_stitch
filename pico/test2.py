import sys

print("PICO_READY")

while True:
    print("PICO_READY")

    while True:
        command = input().strip()

        if command == "PING":
            print("PONG")
        elif command:
            print(f"RECEIVED:{command}")