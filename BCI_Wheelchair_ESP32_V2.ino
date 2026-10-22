/*
  BCI-Controlled Toy Wheelchair - ESP32 Firmware V2
  -------------------------------------------------
  Command sources:
    1. WiFi HTTP: /move?dir=forward
    2. Bluetooth Classic SPP: forward\n
    3. USB Serial: forward\n

  Safety:
    - 3-second command watchdog -> STOP
    - HC-SR04 blocks FORWARD below 15 cm
    - No potentiometer/autonomous fallback
    - Motors start in STOP state

  IMPORTANT:
    This firmware keeps the exact pin map from the original project.
    HC-SR04 ECHO must be reduced to 3.3 V before GPIO 18.
*/

#include <WiFi.h>
#include <WebServer.h>
#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled. Select an ESP32 board with Bluetooth Classic/SPP support.
#endif

// ===================== CONFIGURATION =====================

const char* WIFI_SSID      = "WheelChair";
const char* WIFI_PASSWORD  = "neuromove";
const char* BT_DEVICE_NAME = "NeuroMove_ESP32";

const uint16_t SERVER_PORT = 80;

const unsigned long WIFI_CONNECT_TIMEOUT_MS = 8000;
const unsigned long COMMAND_TIMEOUT_MS      = 3000;
const unsigned long SENSOR_INTERVAL_MS     = 100;

const float OBSTACLE_THRESHOLD_CM = 15.0f;

const int MOTOR_SPEED = 200;   // 0-255
const int PWM_FREQ    = 1000;
const int PWM_RES     = 8;

// Set true only if you want the ESP32 WiFi radio kept awake.
// This can improve responsiveness but uses more power.
const bool WIFI_NO_SLEEP = true;

// ===================== EXACT ORIGINAL PIN MAP =====================

const int PIN_IN1 = 14;
const int PIN_IN2 = 27;
const int PIN_IN3 = 26;
const int PIN_IN4 = 25;

const int PIN_ENA = 33;
const int PIN_ENB = 32;

const int PIN_TRIG = 5;
const int PIN_ECHO = 18;

const int PIN_POT = 34;   // Kept wired/available, but NOT used as fallback.

// ===================== GLOBALS =====================

WebServer server(SERVER_PORT);
BluetoothSerial SerialBT;

bool wifiConnected = false;
bool httpServerStarted = false;

String lastReceivedCommand = "stop";
String currentMotorState   = "stop";
String lastCommandSource   = "none";

unsigned long lastCommandTime   = 0;
unsigned long lastSensorReadTime = 0;

float lastDistanceCm = 999.0f;
bool obstacleDetected = false;

// Non-blocking Bluetooth line parser.
char btBuffer[32];
size_t btBufferIndex = 0;

// Non-blocking USB Serial line parser.
char serialBuffer[32];
size_t serialBufferIndex = 0;

// ===================== FORWARD DECLARATIONS =====================

void connectToWiFi();
void setupWebServer();
void maintainWiFi();

void handleRoot();
void handleMoveRequest();

void pollBluetooth();
void pollUSBSerial();

void processIncomingCommand(const String& direction, const String& source);
bool isValidCommand(const String& command);

String determineActiveCommand();

void readUltrasonicSensor();
void checkObstacleOverride();

void executeCommand(const String& command);

void moveForward();
void moveBackward();
void turnLeft();
void turnRight();
void stopMotors();

void processBluetoothCharacter(char c);
void processSerialCharacter(char c);

// ===================== SETUP =====================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("========================================");
  Serial.println(" BCI Wheelchair - ESP32 Firmware V2");
  Serial.println(" WiFi + Bluetooth Classic + USB Serial");
  Serial.println("========================================");

  // Motor direction pins
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);

  // PWM
  bool pwmA = ledcAttach(PIN_ENA, PWM_FREQ, PWM_RES);
  bool pwmB = ledcAttach(PIN_ENB, PWM_FREQ, PWM_RES);

  if (!pwmA || !pwmB) {
    Serial.println("[ERROR] PWM attach failed!");
  }

  // Ultrasonic
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  digitalWrite(PIN_TRIG, LOW);

  // Potentiometer pin remains configured, but is NOT used for motion.
  pinMode(PIN_POT, INPUT);

  // Safety: motors OFF before communication starts.
  stopMotors();

  // Start Bluetooth Classic SPP.
  SerialBT.begin(BT_DEVICE_NAME);
  Serial.print("[BLUETOOTH] Started: ");
  Serial.println(BT_DEVICE_NAME);

  // Start WiFi.
  connectToWiFi();

  if (wifiConnected) {
    setupWebServer();
    server.begin();
    httpServerStarted = true;

    Serial.print("[SERVER] HTTP server: http://");
    Serial.print(WiFi.localIP());
    Serial.println("/");

    Serial.println("[SERVER] Command example:");
    Serial.print("         http://");
    Serial.print(WiFi.localIP());
    Serial.println("/move?dir=forward");
  } else {
    Serial.println("[WIFI] Unavailable. Bluetooth + USB Serial remain active.");
  }

  Serial.println();
  Serial.println("[INFO] Valid commands:");
  Serial.println("       forward");
  Serial.println("       backward");
  Serial.println("       left");
  Serial.println("       right");
  Serial.println("       stop");
  Serial.println();
  Serial.println("[SAFETY] No command for 3 seconds -> STOP");
  Serial.println("[SAFETY] Forward blocked below 15 cm");
  Serial.println("[INFO] Ready.");
}

// ===================== MAIN LOOP =====================

void loop() {
  if (wifiConnected && httpServerStarted) {
    server.handleClient();
  }

  pollBluetooth();
  pollUSBSerial();

  // Keep WiFi state healthy without blocking the control loop.
  maintainWiFi();

  unsigned long now = millis();

  if (now - lastSensorReadTime >= SENSOR_INTERVAL_MS) {
    lastSensorReadTime = now;

    readUltrasonicSensor();
    checkObstacleOverride();
  }

  String command = determineActiveCommand();
  executeCommand(command);

  delay(2);
}

// ===================== WIFI =====================

void connectToWiFi() {
  Serial.print("[WIFI] Connecting to: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);

  if (WIFI_NO_SLEEP) {
    WiFi.setSleep(false);
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
    delay(100);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;

    Serial.print("[WIFI] Connected. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    wifiConnected = false;
    Serial.println("[WIFI] Connection failed.");
  }
}

void maintainWiFi() {
  static unsigned long lastWiFiCheck = 0;
  static unsigned long lastReconnectAttempt = 0;

  unsigned long now = millis();

  if (now - lastWiFiCheck < 1000) {
    return;
  }

  lastWiFiCheck = now;

  bool currentlyConnected = (WiFi.status() == WL_CONNECTED);

  if (currentlyConnected) {
    if (!wifiConnected) {
      wifiConnected = true;

      Serial.print("[WIFI] Reconnected. IP: ");
      Serial.println(WiFi.localIP());

      if (!httpServerStarted) {
        setupWebServer();
        server.begin();
        httpServerStarted = true;
        Serial.println("[SERVER] HTTP server restarted.");
      }
    }
    return;
  }

  // Do not repeatedly call WiFi.begin every millisecond.
  if (now - lastReconnectAttempt >= 10000) {
    lastReconnectAttempt = now;

    if (wifiConnected) {
      Serial.println("[WIFI] Connection lost. Motor safety watchdog remains active.");
    }

    wifiConnected = false;

    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.println("[WIFI] Reconnect attempt started...");
  }
}

void setupWebServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/move", HTTP_GET, handleMoveRequest);

  server.onNotFound([]() {
    server.send(
      404,
      "text/plain",
      "404: Use /move?dir=forward|backward|left|right|stop"
    );
  });
}

void handleRoot() {
  String response = "{";
  response += "\"status\":\"ok\",";
  response += "\"command\":\"" + lastReceivedCommand + "\",";
  response += "\"motor\":\"" + currentMotorState + "\",";
  response += "\"source\":\"" + lastCommandSource + "\",";
  response += "\"distance_cm\":" + String(lastDistanceCm, 1) + ",";
  response += "\"obstacle\":" + String(obstacleDetected ? "true" : "false") + ",";
  response += "\"wifi\":\"" + WiFi.localIP().toString() + "\",";
  response += "\"bluetooth\":\"" + String(BT_DEVICE_NAME) + "\"";
  response += "}";

  server.send(200, "application/json", response);
}

void handleMoveRequest() {
  if (!server.hasArg("dir")) {
    server.send(400, "text/plain", "Missing 'dir' parameter.");
    return;
  }

  String direction = server.arg("dir");
  direction.toLowerCase();
  direction.trim();

  if (!isValidCommand(direction)) {
    server.send(
      400,
      "application/json",
      "{\"status\":\"error\",\"reason\":\"invalid_command\"}"
    );
    return;
  }

  processIncomingCommand(direction, "wifi");

  if (obstacleDetected && direction == "forward") {
    String response = "{\"status\":\"blocked\",\"command\":\"forward\",";
    response += "\"reason\":\"obstacle_detected\",\"distance_cm\":";
    response += String(lastDistanceCm, 1);
    response += "}";

    server.send(200, "application/json", response);
    return;
  }

  String response = "{\"status\":\"ok\",\"command\":\"";
  response += direction;
  response += "\"}";

  server.send(200, "application/json", response);
}

// ===================== BLUETOOTH =====================

void pollBluetooth() {
  while (SerialBT.available()) {
    char c = (char)SerialBT.read();
    processBluetoothCharacter(c);
  }
}

void processBluetoothCharacter(char c) {
  // Accept either newline style.
  if (c == '\n' || c == '\r') {
    if (btBufferIndex == 0) {
      return;
    }

    btBuffer[btBufferIndex] = '\0';

    String command = String(btBuffer);
    command.toLowerCase();
    command.trim();

    btBufferIndex = 0;

    if (command.length() > 0) {
      if (isValidCommand(command)) {
        processIncomingCommand(command, "bluetooth");
        SerialBT.println("OK: " + command);
      } else {
        SerialBT.println("ERROR: invalid command");
        Serial.println("[BLUETOOTH] Invalid command ignored.");
      }
    }

    return;
  }

  // Ignore other control characters.
  if ((unsigned char)c < 32) {
    return;
  }

  // Prevent buffer overflow.
  if (btBufferIndex < sizeof(btBuffer) - 1) {
    btBuffer[btBufferIndex++] = c;
  } else {
    btBufferIndex = 0;
    SerialBT.println("ERROR: command too long");
  }
}

// ===================== USB SERIAL =====================

void pollUSBSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    processSerialCharacter(c);
  }
}

void processSerialCharacter(char c) {
  if (c == '\n' || c == '\r') {
    if (serialBufferIndex == 0) {
      return;
    }

    serialBuffer[serialBufferIndex] = '\0';

    String command = String(serialBuffer);
    command.toLowerCase();
    command.trim();

    serialBufferIndex = 0;

    if (command.length() > 0) {
      processIncomingCommand(command, "serial");
    }

    return;
  }

  if ((unsigned char)c < 32) {
    return;
  }

  if (serialBufferIndex < sizeof(serialBuffer) - 1) {
    serialBuffer[serialBufferIndex++] = c;
  } else {
    serialBufferIndex = 0;
    Serial.println("[SERIAL] Command too long.");
  }
}

// ===================== COMMAND HANDLING =====================

bool isValidCommand(const String& command) {
  return command == "forward" ||
         command == "backward" ||
         command == "left" ||
         command == "right" ||
         command == "stop";
}

void processIncomingCommand(const String& direction, const String& source) {
  if (!isValidCommand(direction)) {
    Serial.print("[");
    Serial.print(source);
    Serial.print("] Invalid command ignored: ");
    Serial.println(direction);
    return;
  }

  lastReceivedCommand = direction;
  lastCommandTime = millis();
  lastCommandSource = source;

  Serial.print("[");
  Serial.print(source);
  Serial.print("] Command received: ");
  Serial.println(direction);

  if (obstacleDetected && direction == "forward") {
    Serial.println("[SAFETY] Forward command received, but obstacle is detected.");
  }
}

// ===================== COMMAND WATCHDOG =====================

String determineActiveCommand() {
  if (millis() - lastCommandTime < COMMAND_TIMEOUT_MS) {
    return lastReceivedCommand;
  }

  // SAFETY: never use the potentiometer as an autonomous fallback.
  return "stop";
}

// ===================== ULTRASONIC =====================

void readUltrasonicSensor() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);

  digitalWrite(PIN_TRIG, LOW);

  // 20 ms is enough for roughly 3.4 m maximum range and
  // reduces worst-case blocking compared with the old 30 ms timeout.
  unsigned long duration = pulseIn(PIN_ECHO, HIGH, 20000);

  if (duration == 0) {
    lastDistanceCm = 999.0f;
  } else {
    lastDistanceCm = (duration * 0.0343f) / 2.0f;
  }

  static unsigned long lastPrintTime = 0;

  if (millis() - lastPrintTime >= 1000) {
    lastPrintTime = millis();

    Serial.print("[SENSOR] Distance: ");
    Serial.print(lastDistanceCm, 1);
    Serial.println(" cm");
  }
}

void checkObstacleOverride() {
  bool wasDetected = obstacleDetected;

  obstacleDetected =
    (lastDistanceCm > 0.0f &&
     lastDistanceCm < OBSTACLE_THRESHOLD_CM);

  if (obstacleDetected && !wasDetected) {
    Serial.println("*** [SAFETY] OBSTACLE DETECTED - FORWARD BLOCKED ***");

    if (currentMotorState == "forward") {
      stopMotors();
      currentMotorState = "stop";
    }
  }

  if (!obstacleDetected && wasDetected) {
    Serial.println("*** [SAFETY] Path clear ***");
  }
}

// ===================== MOTOR CONTROL =====================

void executeCommand(const String& command) {
  // Ultrasonic safety gate applies ONLY to forward.
  // Turning and reversing remain available to let the robot escape
  // from an obstacle.
  if (obstacleDetected && command == "forward") {
    if (currentMotorState != "stop") {
      stopMotors();
      currentMotorState = "stop";
    }
    return;
  }

  if (command == currentMotorState) {
    return;
  }

  Serial.print("[MOTOR] Executing: ");
  Serial.println(command);

  if (command == "forward") {
    moveForward();
  } else if (command == "backward") {
    moveBackward();
  } else if (command == "left") {
    turnLeft();
  } else if (command == "right") {
    turnRight();
  } else {
    stopMotors();
  }

  currentMotorState = command;
}

void moveForward() {
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);

  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);

  ledcWrite(PIN_ENA, MOTOR_SPEED);
  ledcWrite(PIN_ENB, MOTOR_SPEED);
}

void moveBackward() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);

  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);

  ledcWrite(PIN_ENA, MOTOR_SPEED);
  ledcWrite(PIN_ENB, MOTOR_SPEED);
}

void turnLeft() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);

  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);

  ledcWrite(PIN_ENA, MOTOR_SPEED);
  ledcWrite(PIN_ENB, MOTOR_SPEED);
}

void turnRight() {
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);

  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);

  ledcWrite(PIN_ENA, MOTOR_SPEED);
  ledcWrite(PIN_ENB, MOTOR_SPEED);
}

void stopMotors() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);

  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);

  ledcWrite(PIN_ENA, 0);
  ledcWrite(PIN_ENB, 0);
}
