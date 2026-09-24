// GreggSpeak ESP32 feeder controller
// ESP32 DevKit V1 + IBT-2 / BTS7960 motor driver + tray IR sensor
// USB serial protocol: newline-terminated ASCII commands at 115200 baud

#define R_EN 25
#define L_EN 26
#define RPWM 27
#define LPWM 12
#define IR_SENSOR 33

#define PWM_FREQ 20000
#define PWM_RESOLUTION 8

const bool IR_ACTIVE_LOW = true;
const int DEFAULT_SPEED = 150;
const int MAX_FEED_MS = 12000;
const int SENSOR_DEBOUNCE_MS = 25;

bool motorRunning = false;
unsigned long motorStartMs = 0;
unsigned long requestedFeedMs = 0;
String currentJobId = "";
int defaultSpeed = DEFAULT_SPEED;

String inputLine = "";

void setup() {
  Serial.begin(115200);

  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(IR_SENSOR, INPUT_PULLUP);

  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);

  ledcAttach(RPWM, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(LPWM, PWM_FREQ, PWM_RESOLUTION);

  stopMotor();
  Serial.println("GREGGSPEAK_FEEDER_READY");
}

void loop() {
  readSerialCommands();
  finishFeedWhenDue();
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char incoming = (char)Serial.read();
    if (incoming == '\n' || incoming == '\r') {
      inputLine.trim();
      if (inputLine.length() > 0) {
        handleCommand(inputLine);
      }
      inputLine = "";
    } else {
      inputLine += incoming;
      if (inputLine.length() > 96) {
        inputLine = "";
        Serial.println("ERR BAD_COMMAND");
      }
    }
  }
}

void handleCommand(String line) {
  String command = tokenAt(line, 0);
  command.toUpperCase();

  if (command == "PING") {
    Serial.println("OK PONG");
    return;
  }

  if (command == "STATUS") {
    printStatus();
    return;
  }

  if (command == "SENSOR?") {
    printSensorStatus();
    return;
  }

  if (command == "STOP") {
    stopMotor();
    motorRunning = false;
    currentJobId = "";
    requestedFeedMs = 0;
    Serial.println("OK STOPPED");
    return;
  }

  if (command == "SET_SPEED") {
    int speedValue = tokenAt(line, 1).toInt();
    if (speedValue < 0 || speedValue > 255) {
      Serial.println("ERR LIMIT_EXCEEDED");
      return;
    }
    defaultSpeed = speedValue;
    Serial.print("OK SPEED ");
    Serial.println(defaultSpeed);
    return;
  }

  if (command == "FEED") {
    handleFeedCommand(line);
    return;
  }

  Serial.println("ERR BAD_COMMAND");
}

void handleFeedCommand(String line) {
  if (motorRunning) {
    Serial.println("ERR BUSY");
    return;
  }

  int durationMs = tokenAt(line, 1).toInt();
  int speedValue = tokenAt(line, 2).toInt();
  String jobId = tokenAt(line, 3);

  if (jobId.length() == 0) {
    jobId = "JOB";
  }

  if (durationMs <= 0 || durationMs > MAX_FEED_MS || speedValue < 0 || speedValue > 255) {
    Serial.println("ERR LIMIT_EXCEEDED");
    return;
  }

  requestedFeedMs = (unsigned long)durationMs;
  currentJobId = jobId;
  motorStartMs = millis();
  motorRunning = true;
  forwardMotor(speedValue);

  Serial.print("OK FEED_STARTED ");
  Serial.println(currentJobId);
}

void finishFeedWhenDue() {
  if (!motorRunning) {
    return;
  }

  unsigned long elapsed = millis() - motorStartMs;
  if (elapsed < requestedFeedMs) {
    return;
  }

  stopMotor();
  motorRunning = false;

  Serial.print("OK FEED_DONE ");
  Serial.print(currentJobId);
  Serial.print(" ELAPSED=");
  Serial.print(elapsed);
  Serial.print(" PAPER=");
  Serial.println(readPaperPresent() ? 1 : 0);

  currentJobId = "";
  requestedFeedMs = 0;
}

void printStatus() {
  Serial.print("OK STATUS PAPER=");
  Serial.print(readPaperPresent() ? 1 : 0);
  Serial.print(" MOTOR=");
  Serial.print(motorRunning ? 1 : 0);
  Serial.print(" STATE=");
  Serial.println(motorRunning ? "FEEDING" : "IDLE");
}

void printSensorStatus() {
  if (readPaperPresent()) {
    Serial.println("OK PAPER_PRESENT");
  } else {
    Serial.println("OK NO_PAPER");
  }
}

bool readPaperPresent() {
  int highCount = 0;
  int lowCount = 0;
  unsigned long start = millis();

  while (millis() - start < SENSOR_DEBOUNCE_MS) {
    if (digitalRead(IR_SENSOR) == HIGH) {
      highCount++;
    } else {
      lowCount++;
    }
    delay(1);
  }

  bool rawHigh = highCount >= lowCount;
  return IR_ACTIVE_LOW ? !rawHigh : rawHigh;
}

void forwardMotor(int speedValue) {
  speedValue = constrain(speedValue, 0, 255);
  ledcWrite(RPWM, 0);
  ledcWrite(LPWM, speedValue);
}

void reverseMotor(int speedValue) {
  speedValue = constrain(speedValue, 0, 255);
  ledcWrite(RPWM, speedValue);
  ledcWrite(LPWM, 0);
}

void stopMotor() {
  ledcWrite(RPWM, 0);
  ledcWrite(LPWM, 0);
}

String tokenAt(String line, int index) {
  line.trim();
  int currentIndex = 0;
  int tokenStart = 0;

  for (int i = 0; i <= line.length(); i++) {
    if (i == line.length() || line.charAt(i) == ' ') {
      if (currentIndex == index) {
        return line.substring(tokenStart, i);
      }
      currentIndex++;
      while (i + 1 < line.length() && line.charAt(i + 1) == ' ') {
        i++;
      }
      tokenStart = i + 1;
    }
  }

  return "";
}
