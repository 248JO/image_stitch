#include <Arduino.h>
#include <stdlib.h>
#include <errno.h>

// =====================================================
// X AXIS: Motors 1 and 2 move together
// =====================================================
const int X_DIR_PIN  = 16;
const int X_PUL1_PIN = 17;
const int X_PUL2_PIN = 4;

const int X_MIN_SWITCH = 2;   // GP2 = X home
const int X_MAX_SWITCH = 3;   // GP3 = X maximum

// =====================================================
// Y AXIS: Motor 3
// =====================================================
const int Y_DIR_PIN = 15;
const int Y_PUL_PIN = 14;

const int Y_MIN_SWITCH = 7;   // GP7 = Y home
const int Y_MAX_SWITCH = 8;   // GP8 = Y maximum

// =====================================================
// Z AXIS: Motor 4
// =====================================================
const int Z_PUL_PIN = 9;
const int Z_DIR_PIN = 10;

const int Z_MIN_SWITCH = 11;  // GP11 = Z home
const int Z_MAX_SWITCH = 12;  // GP12 = Z maximum

// =====================================================
// Motion settings
// =====================================================

// Must match the TB6600 DIP-switch pulse/revolution setting.
const int PULSES_PER_REV = 3200;

// Start low and increase later only if motion stays reliable.
const int RPM = 150;

const unsigned long PULSE_FREQUENCY =
    (RPM * PULSES_PER_REV) / 60;

const unsigned long STEP_PERIOD_US =
    1000000UL / PULSE_FREQUENCY;

const unsigned long PULSE_WIDTH_US = 10;

const unsigned long TRAVEL_TIMEOUT_MS = 30000;
const unsigned long INPUT_DELAY_MS = 3000;

// Change HIGH/LOW if an axis moves away from the intended switch.
const int X_DIRECTION_TO_MIN = HIGH;
const int X_DIRECTION_TO_MAX = LOW;

const int Y_DIRECTION_TO_MIN = HIGH;
const int Y_DIRECTION_TO_MAX = LOW;

const int Z_DIRECTION_TO_MIN = HIGH;
const int Z_DIRECTION_TO_MAX = LOW;

// =====================================================
// Measured travel time between limit switches
// =====================================================
unsigned long xMaxTimeMs = 0;
unsigned long yMaxTimeMs = 0;
unsigned long zMaxTimeMs = 0;

// Estimated current time-position from home.
// 0 = minimum/home switch.
// maxTime = maximum switch.
long currentXTimeMs = 0;
long currentYTimeMs = 0;
long currentZTimeMs = 0;

String inputBuffer = "";

// NPN photoelectric switch: LOW = triggered
bool isTriggered(int switchPin) {
  return digitalRead(switchPin) == LOW;
}

void stopAll() {
  digitalWrite(X_PUL1_PIN, LOW);
  digitalWrite(X_PUL2_PIN, LOW);
  digitalWrite(Y_PUL_PIN, LOW);
  digitalWrite(Z_PUL_PIN, LOW);
}

void stepX() {
  digitalWrite(X_PUL1_PIN, HIGH);
  digitalWrite(X_PUL2_PIN, HIGH);
  delayMicroseconds(PULSE_WIDTH_US);

  digitalWrite(X_PUL1_PIN, LOW);
  digitalWrite(X_PUL2_PIN, LOW);
}

void stepY() {
  digitalWrite(Y_PUL_PIN, HIGH);
  delayMicroseconds(PULSE_WIDTH_US);
  digitalWrite(Y_PUL_PIN, LOW);
}

void stepZ() {
  digitalWrite(Z_PUL_PIN, HIGH);
  delayMicroseconds(PULSE_WIDTH_US);
  digitalWrite(Z_PUL_PIN, LOW);
}

// =====================================================
// Move toward one specific limit switch
// Used for homing and initial travel-time measurement.
// =====================================================
bool moveXUntilSwitch(int targetSwitch, int direction,
                      unsigned long &elapsedMs) {
  digitalWrite(X_DIR_PIN, direction);
  delay(10);

  unsigned long startTime = millis();

  while (!isTriggered(targetSwitch)) {
    if (millis() - startTime > TRAVEL_TIMEOUT_MS) {
      stopAll();
      Serial.println("ERROR: X timeout.");
      return false;
    }

    stepX();
    delayMicroseconds(STEP_PERIOD_US - PULSE_WIDTH_US);
  }

  elapsedMs = millis() - startTime;

  digitalWrite(X_PUL1_PIN, LOW);
  digitalWrite(X_PUL2_PIN, LOW);
  return true;
}

bool moveYUntilSwitch(int targetSwitch, int direction,
                      unsigned long &elapsedMs) {
  digitalWrite(Y_DIR_PIN, direction);
  delay(10);

  unsigned long startTime = millis();

  while (!isTriggered(targetSwitch)) {
    if (millis() - startTime > TRAVEL_TIMEOUT_MS) {
      stopAll();
      Serial.println("ERROR: Y timeout.");
      return false;
    }

    stepY();
    delayMicroseconds(STEP_PERIOD_US - PULSE_WIDTH_US);
  }

  elapsedMs = millis() - startTime;

  digitalWrite(Y_PUL_PIN, LOW);
  return true;
}

bool moveZUntilSwitch(int targetSwitch, int direction,
                      unsigned long &elapsedMs) {
  digitalWrite(Z_DIR_PIN, direction);
  delay(10);

  unsigned long startTime = millis();

  while (!isTriggered(targetSwitch)) {
    if (millis() - startTime > TRAVEL_TIMEOUT_MS) {
      stopAll();
      Serial.println("ERROR: Z timeout.");
      return false;
    }

    stepZ();
    delayMicroseconds(STEP_PERIOD_US - PULSE_WIDTH_US);
  }

  elapsedMs = millis() - startTime;

  digitalWrite(Z_PUL_PIN, LOW);
  return true;
}

// =====================================================
// Home each axis
// =====================================================
bool homeX() {
  unsigned long ignoredTime = 0;

  Serial.println("Homing X to GP2...");
  if (!moveXUntilSwitch(X_MIN_SWITCH, X_DIRECTION_TO_MIN, ignoredTime)) {
    return false;
  }

  currentXTimeMs = 0;
  Serial.println("X homed.");
  delay(500);
  return true;
}

bool homeY() {
  unsigned long ignoredTime = 0;

  Serial.println("Homing Y to GP7...");
  if (!moveYUntilSwitch(Y_MIN_SWITCH, Y_DIRECTION_TO_MIN, ignoredTime)) {
    return false;
  }

  currentYTimeMs = 0;
  Serial.println("Y homed.");
  delay(500);
  return true;
}

bool homeZ() {
  unsigned long ignoredTime = 0;

  Serial.println("Homing Z to GP11...");
  if (!moveZUntilSwitch(Z_MIN_SWITCH, Z_DIRECTION_TO_MIN, ignoredTime)) {
    return false;
  }

  currentZTimeMs = 0;
  Serial.println("Z homed.");
  delay(500);
  return true;
}

// =====================================================
// Measure the travel time from home/minimum to maximum,
// then return the axis home.
// =====================================================
bool calibrateTravelTimes() {
  if (!homeX()) return false;

  Serial.println("Measuring X: GP2 to GP3...");
  if (!moveXUntilSwitch(X_MAX_SWITCH, X_DIRECTION_TO_MAX, xMaxTimeMs)) {
    return false;
  }

  if (xMaxTimeMs == 0) {
    Serial.println("ERROR: X travel time measured as zero.");
    return false;
  }

  Serial.print("X full travel time: ");
  Serial.print(xMaxTimeMs);
  Serial.println(" ms");

  if (!homeX()) return false;

  if (!homeY()) return false;

  Serial.println("Measuring Y: GP7 to GP8...");
  if (!moveYUntilSwitch(Y_MAX_SWITCH, Y_DIRECTION_TO_MAX, yMaxTimeMs)) {
    return false;
  }

  if (yMaxTimeMs == 0) {
    Serial.println("ERROR: Y travel time measured as zero.");
    return false;
  }

  Serial.print("Y full travel time: ");
  Serial.print(yMaxTimeMs);
  Serial.println(" ms");

  if (!homeY()) return false;

  if (!homeZ()) return false;

  Serial.println("Measuring Z: GP11 to GP12...");
  if (!moveZUntilSwitch(Z_MAX_SWITCH, Z_DIRECTION_TO_MAX, zMaxTimeMs)) {
    return false;
  }

  if (zMaxTimeMs == 0) {
    Serial.println("ERROR: Z travel time measured as zero.");
    return false;
  }

  Serial.print("Z full travel time: ");
  Serial.print(zMaxTimeMs);
  Serial.println(" ms");

  if (!homeZ()) return false;

  return true;
}

// =====================================================
// Time movement with both limit switches protected.
// An axis stops immediately if either switch is touched.
// =====================================================
bool moveXForTime(long signedTimeMs) {
  if (signedTimeMs == 0) return true;

  bool towardMax = signedTimeMs > 0;
  unsigned long requestedTimeMs = labs(signedTimeMs);

  digitalWrite(X_DIR_PIN,
               towardMax ? X_DIRECTION_TO_MAX : X_DIRECTION_TO_MIN);
  delay(10);

  // Allows departure from a switch if the axis begins at home/max.
  bool minWasClear = !isTriggered(X_MIN_SWITCH);
  bool maxWasClear = !isTriggered(X_MAX_SWITCH);

  unsigned long startTime = millis();

  while (millis() - startTime < requestedTimeMs) {
    // Stop immediately if either limit is touched after being released.
    if ((minWasClear && isTriggered(X_MIN_SWITCH)) ||
        (maxWasClear && isTriggered(X_MAX_SWITCH))) {

      digitalWrite(X_PUL1_PIN, LOW);
      digitalWrite(X_PUL2_PIN, LOW);

      if (isTriggered(X_MIN_SWITCH)) {
        currentXTimeMs = 0;
        Serial.println("X minimum switch touched. X stopped.");
      } else {
        currentXTimeMs = xMaxTimeMs;
        Serial.println("X maximum switch touched. X stopped.");
      }

      return true;
    }

    stepX();

    if (!isTriggered(X_MIN_SWITCH)) minWasClear = true;
    if (!isTriggered(X_MAX_SWITCH)) maxWasClear = true;

    delayMicroseconds(STEP_PERIOD_US - PULSE_WIDTH_US);
  }

  digitalWrite(X_PUL1_PIN, LOW);
  digitalWrite(X_PUL2_PIN, LOW);

  currentXTimeMs += signedTimeMs;
  return true;
}

bool moveYForTime(long signedTimeMs) {
  if (signedTimeMs == 0) return true;

  bool towardMax = signedTimeMs > 0;
  unsigned long requestedTimeMs = labs(signedTimeMs);

  digitalWrite(Y_DIR_PIN,
               towardMax ? Y_DIRECTION_TO_MAX : Y_DIRECTION_TO_MIN);
  delay(10);

  bool minWasClear = !isTriggered(Y_MIN_SWITCH);
  bool maxWasClear = !isTriggered(Y_MAX_SWITCH);

  unsigned long startTime = millis();

  while (millis() - startTime < requestedTimeMs) {
    if ((minWasClear && isTriggered(Y_MIN_SWITCH)) ||
        (maxWasClear && isTriggered(Y_MAX_SWITCH))) {

      digitalWrite(Y_PUL_PIN, LOW);

      if (isTriggered(Y_MIN_SWITCH)) {
        currentYTimeMs = 0;
        Serial.println("Y minimum switch touched. Y stopped.");
      } else {
        currentYTimeMs = yMaxTimeMs;
        Serial.println("Y maximum switch touched. Y stopped.");
      }

      return true;
    }

    stepY();

    if (!isTriggered(Y_MIN_SWITCH)) minWasClear = true;
    if (!isTriggered(Y_MAX_SWITCH)) maxWasClear = true;

    delayMicroseconds(STEP_PERIOD_US - PULSE_WIDTH_US);
  }

  digitalWrite(Y_PUL_PIN, LOW);

  currentYTimeMs += signedTimeMs;
  return true;
}

bool moveZForTime(long signedTimeMs) {
  if (signedTimeMs == 0) return true;

  bool towardMax = signedTimeMs > 0;
  unsigned long requestedTimeMs = labs(signedTimeMs);

  digitalWrite(Z_DIR_PIN,
               towardMax ? Z_DIRECTION_TO_MAX : Z_DIRECTION_TO_MIN);
  delay(10);

  bool minWasClear = !isTriggered(Z_MIN_SWITCH);
  bool maxWasClear = !isTriggered(Z_MAX_SWITCH);

  unsigned long startTime = millis();

  while (millis() - startTime < requestedTimeMs) {
    if ((minWasClear && isTriggered(Z_MIN_SWITCH)) ||
        (maxWasClear && isTriggered(Z_MAX_SWITCH))) {

      digitalWrite(Z_PUL_PIN, LOW);

      if (isTriggered(Z_MIN_SWITCH)) {
        currentZTimeMs = 0;
        Serial.println("Z minimum switch touched. Z stopped.");
      } else {
        currentZTimeMs = zMaxTimeMs;
        Serial.println("Z maximum switch touched. Z stopped.");
      }

      return true;
    }

    stepZ();

    if (!isTriggered(Z_MIN_SWITCH)) minWasClear = true;
    if (!isTriggered(Z_MAX_SWITCH)) maxWasClear = true;

    delayMicroseconds(STEP_PERIOD_US - PULSE_WIDTH_US);
  }

  digitalWrite(Z_PUL_PIN, LOW);

  currentZTimeMs += signedTimeMs;
  return true;
}

void printPrompt() {
  Serial.println();
  Serial.println("========== TIME POSITION ==========");
  Serial.print("X = "); Serial.print(currentXTimeMs);
  Serial.print(" ms, maximum = "); Serial.print(xMaxTimeMs);
  Serial.println(" ms");

  Serial.print("Y = "); Serial.print(currentYTimeMs);
  Serial.print(" ms, maximum = "); Serial.print(yMaxTimeMs);
  Serial.println(" ms");

  Serial.print("Z = "); Serial.print(currentZTimeMs);
  Serial.print(" ms, maximum = "); Serial.print(zMaxTimeMs);
  Serial.println(" ms");

  Serial.println("===================================");
  Serial.println("Enter relative time: X,Y,Z in milliseconds.");
  Serial.println("Example: 1000,-500,0");
  Serial.println("Positive = toward maximum.");
  Serial.println("Negative = toward home.");
  Serial.println("X moves first, then Y, then Z.");
  Serial.print("> ");
}

bool parseSignedTime(const String &text, long &result) {
  if (text.length() == 0) return false;

  errno = 0;
  char *end = nullptr;
  long parsed = strtol(text.c_str(), &end, 10);

  if (errno == ERANGE || *end != '\0') return false;

  result = parsed;
  return true;
}

// =====================================================
// Startup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(5000);

  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(X_PUL1_PIN, OUTPUT);
  pinMode(X_PUL2_PIN, OUTPUT);

  pinMode(Y_DIR_PIN, OUTPUT);
  pinMode(Y_PUL_PIN, OUTPUT);

  pinMode(Z_DIR_PIN, OUTPUT);
  pinMode(Z_PUL_PIN, OUTPUT);

  // NPN sensors: external 10 kΩ pull-ups to 3.3 V required.
  pinMode(X_MIN_SWITCH, INPUT);
  pinMode(X_MAX_SWITCH, INPUT);
  pinMode(Y_MIN_SWITCH, INPUT);
  pinMode(Y_MAX_SWITCH, INPUT);
  pinMode(Z_MIN_SWITCH, INPUT);
  pinMode(Z_MAX_SWITCH, INPUT);

  stopAll();

  Serial.println();
  Serial.println("Starting home and travel-time calibration...");

  if (!calibrateTravelTimes()) {
    stopAll();
    Serial.println("ERROR: Calibration failed. Motors stopped.");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("Calibration complete. All axes are home.");
  Serial.println("Waiting 3 seconds before first input...");
  delay(INPUT_DELAY_MS);

  printPrompt();
}

// =====================================================
// Input format: X,Y,Z in signed milliseconds
// =====================================================
void loop() {
  while (Serial.available() > 0) {
    char character = Serial.read();

    if (character != '\r' && character != '\n') {
      if (inputBuffer.length() < 63) {
        inputBuffer += character;
        Serial.print(character);
      }
      continue;
    }

    if (inputBuffer.length() == 0) continue;

    Serial.println();

    String input = inputBuffer;
    inputBuffer = "";

    input.trim();
    input.replace("(", "");
    input.replace(")", "");
    input.trim();

    int comma1 = input.indexOf(',');
    int comma2 = input.indexOf(',', comma1 + 1);

    if (comma1 == -1 || comma2 == -1 ||
        input.indexOf(',', comma2 + 1) != -1) {
      Serial.println("Invalid format. Example: 1000,-500,0");
      delay(INPUT_DELAY_MS);
      printPrompt();
      continue;
    }

    String xText = input.substring(0, comma1);
    String yText = input.substring(comma1 + 1, comma2);
    String zText = input.substring(comma2 + 1);

    xText.trim();
    yText.trim();
    zText.trim();

    long xTimeMs = 0;
    long yTimeMs = 0;
    long zTimeMs = 0;

    if (!parseSignedTime(xText, xTimeMs) ||
        !parseSignedTime(yText, yTimeMs) ||
        !parseSignedTime(zText, zTimeMs)) {

      Serial.println("Invalid input. Use signed whole milliseconds.");

    } else {
      // Reject command before moving if the final estimated
      // time-position would exceed a measured end limit.
      long targetX = currentXTimeMs + xTimeMs;
      long targetY = currentYTimeMs + yTimeMs;
      long targetZ = currentZTimeMs + zTimeMs;

      if (targetX < 0 || targetX > (long)xMaxTimeMs ||
          targetY < 0 || targetY > (long)yMaxTimeMs ||
          targetZ < 0 || targetZ > (long)zMaxTimeMs) {

        Serial.println("ERROR: Requested final time-position is outside travel range.");

      } else {
        Serial.println("Moving X, then Y, then Z...");

        moveXForTime(xTimeMs);
        moveYForTime(yTimeMs);
        moveZForTime(zTimeMs);

        stopAll();
        Serial.println("Movement complete.");
      }
    }

    Serial.println("Waiting 3 seconds before next input...");
    delay(INPUT_DELAY_MS);

    printPrompt();
  }
}