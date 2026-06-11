/*
 * ═══════════════════════════════════════════════════════════════
 *  SmartFarm IoT — Final Firmware v3.0
 * ═══════════════════════════════════════════════════════════════
 *  Sensors  : DHT22, LDR, Soil Moisture, Ultrasonic HC-SR04
 *  Actuators: Relay x4, Servo Motor
 *  Protocol : HTTP GET → Flask (Render cloud)
 *  Interval : Every 15 seconds
 * ───────────────────────────────────────────────────────────────
 *  PIN MAP:
 *    LDR          → GPIO 34 (analog input)
 *    DHT22        → GPIO 4  (digital)
 *    Ultrasonic   → TRIG GPIO 27 | ECHO GPIO 33
 *    Soil Moisture→ GPIO 35 (analog input)
 *    Relay 1      → GPIO 14 (Pump 1 — ultrasonic)
 *    Relay 2      → GPIO 12 (Pump 2 — soil moisture)
 *    Relay 3      → GPIO 13 (Light  — LDR)
 *    Relay 4      → GPIO 15 (Feeder — timer based)
 *    Servo        → GPIO 25 (PWM)
 * ───────────────────────────────────────────────────────────────
 *  Libraries (Arduino Library Manager):
 *    - DHT sensor library  by Adafruit
 *    - ArduinoJson         by Benoit Blanchon
 *    - ESP32Servo          by Kevin Harrington
 * ═══════════════════════════════════════════════════════════════
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ═══════════════════════════════════════════════
//  WiFi & Server Config
// ═══════════════════════════════════════════════
const char* WIFI_SSID     = "..";
const char* WIFI_PASSWORD = "1234demo";
const char* SERVER_URL    = "https://smartfarm-ai-ttjj.onrender.com/upload";

// ═══════════════════════════════════════════════
//  Pin Definitions
// ═══════════════════════════════════════════════
#define LDR_PIN       34
#define DHT_PIN        4
#define TRIG_PIN      27
#define ECHO_PIN      33
#define SOIL_PIN      35
#define RELAY1_PIN    14   // Pump 1     — water tank (ultrasonic)
#define RELAY2_PIN    12   // Pump 2     — soil irrigation (soil moisture)
#define RELAY3_PIN    13   // Grow Light — LDR based
#define RELAY4_PIN    26   // Feeder     — timer based
#define SERVO_PIN     25   // Servo motor

// ═══════════════════════════════════════════════
//  Sensor & Automation Thresholds
// ═══════════════════════════════════════════════
const float TANK_LOW_CM      = 15.0;   // ultrasonic < 10cm → Pump1 ON
const int   SOIL_DRY_VAL     = 2500;   // soil > 2500       → Pump2 ON
const int   LDR_DARK_VAL     = 1500;   // ldr  < 1500       → Light ON

// ═══════════════════════════════════════════════
//  Timing
// ═══════════════════════════════════════════════
const unsigned long SEND_INTERVAL   = 15000UL;  // 15 seconds
const unsigned long FEEDER_INTERVAL = 60000UL;  // feeder every 1 minute
const unsigned long FEEDER_ON_TIME  = 5000UL;   // feeder ON for 5 seconds

unsigned long lastSendTime    = 0;
unsigned long lastFeederTime  = 0;
unsigned long feederStartTime = 0;
bool          feederRunning   = false;

// ═══════════════════════════════════════════════
//  Objects
// ═══════════════════════════════════════════════
#define DHT_TYPE DHT22
DHT   dht(DHT_PIN, DHT_TYPE);
Servo myServo;

// ═══════════════════════════════════════════════
//  Function Prototypes
// ═══════════════════════════════════════════════
void  connectWiFi();
float readUltrasonic();
int   smoothAnalog(int pin, int samples = 10);
void  handleFeeder();
void  handleRelays(int ldr, int soil, float dist);
bool  sendToCloud(float temp, float hum, int ldr,
                  int soil, float dist, int servo_angle,
                  int r1, int r2, int r3, int r4);

// ═══════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(800);

  Serial.println();
  Serial.println("╔══════════════════════════════════════════════╗");
  Serial.println("║   SmartFarm IoT — Final Firmware v3.0        ║");
  Serial.println("║   LDR · DHT22 · Soil · Ultrasonic · Servo    ║");
  Serial.println("║   4 Relays · Cloud · 15s interval            ║");
  Serial.println("╚══════════════════════════════════════════════╝");
  Serial.println();

  // ── Relay pins — all OFF on boot ──────────────
  // HIGH = relay OFF for optocoupler relay modules
  pinMode(RELAY1_PIN, OUTPUT); digitalWrite(RELAY1_PIN, HIGH);
  pinMode(RELAY2_PIN, OUTPUT); digitalWrite(RELAY2_PIN, HIGH);
  pinMode(RELAY3_PIN, OUTPUT); digitalWrite(RELAY3_PIN, HIGH);
  pinMode(RELAY4_PIN, OUTPUT); digitalWrite(RELAY4_PIN, HIGH);

  // ── Ultrasonic ────────────────────────────────
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // ── Servo ─────────────────────────────────────
  ESP32PWM::allocateTimer(0);
  myServo.setPeriodHertz(50);             // standard 50Hz servo
  myServo.attach(SERVO_PIN, 500, 2400);  // min/max pulse width
  myServo.write(0);                       // start at 0°
  delay(500);

  // ── DHT22 ─────────────────────────────────────
  dht.begin();
  delay(2000);  // DHT22 stabilization time

  // ── WiFi ──────────────────────────────────────
  connectWiFi();

  Serial.println("[SETUP] All systems ready.");
  Serial.printf("[SETUP] Sending data every %lu seconds\n",
                SEND_INTERVAL / 1000);
  Serial.printf("[SETUP] Feeder activates every %lu seconds for %lu seconds\n",
                FEEDER_INTERVAL / 1000, FEEDER_ON_TIME / 1000);
  Serial.println();
}

// ═══════════════════════════════════════════════
void loop() {
  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Lost — reconnecting...");
    connectWiFi();
  }

  // ── Feeder timer runs independently ───────────
  handleFeeder();

  unsigned long now = millis();
  if (now - lastSendTime >= SEND_INTERVAL || lastSendTime == 0) {
    lastSendTime = now;

    Serial.println();
    Serial.println("──────────────────────────────────────────────");
    Serial.printf("[CYCLE] %lu ms uptime\n", now);

    // ── Read DHT22 ────────────────────────────
    float temperature = dht.readTemperature();
    float humidity    = dht.readHumidity();

    if (isnan(temperature) || isnan(humidity)) {
      Serial.println("[DHT22] ERROR — check GPIO 4 + pull-up resistor");
      return;
    }

    // ── Read LDR (smoothed) ───────────────────
    int ldrValue = smoothAnalog(LDR_PIN);

    // ── Read Soil Moisture (smoothed) ─────────
    int soilValue = smoothAnalog(SOIL_PIN);

    // ── Read Ultrasonic ───────────────────────
    float distance = readUltrasonic();

    // ── Servo sweep based on soil moisture ────
    // Dry soil → sweep to 90° (open valve or flag)
    // Wet soil → return to 0°
    int servoAngle = feederRunning ? 90 : 0;
    myServo.write(servoAngle);

    // ── Apply relay logic ─────────────────────
    int relay1 = (distance > 0 && distance > TANK_LOW_CM) ? 1 : 0;
    int relay2 = feederRunning ? 1 : 0;   // timer based
    int relay3 = (soilValue > SOIL_DRY_VAL) ? 1 : 0;
    int relay4 = (ldrValue  < LDR_DARK_VAL) ? 1 : 0;

    handleRelays(relay1, relay2, relay3, relay4);

    // ── Print all readings ────────────────────
    Serial.println("[SENSORS]");
    Serial.printf("  Temperature  : %.2f C\n",  temperature);
    Serial.printf("  Humidity     : %.2f %%\n", humidity);
    Serial.printf("  LDR          : %d\n",      ldrValue);
    Serial.printf("  Soil Moisture: %d\n",      soilValue);
    Serial.printf("  Distance     : %.2f cm\n", distance);
    Serial.printf("  Servo Angle  : %d deg\n",  servoAngle);

    Serial.printf("  Relay1 Pump1 : %s  (dist=%.1fcm > %.0fcm → tank low)\n",
              relay1 ? "ON " : "OFF", distance, TANK_LOW_CM);
    Serial.printf("  Relay2 Feeder: %s  (timer — 5s ON every 1min)\n",
              relay2 ? "ON " : "OFF");
    Serial.printf("  Relay3 Pump2 : %s  (soil=%d > %d → dry)\n",
              relay3 ? "ON " : "OFF", soilValue, SOIL_DRY_VAL);
    Serial.printf("  Relay4 Light : %s  (ldr=%d < %d → dark)\n",
              relay4 ? "ON " : "OFF", ldrValue, LDR_DARK_VAL);
    Serial.printf("  Servo Angle  : %d deg (follows feeder timer)\n",
              servoAngle);

    // ── Send to Render cloud ──────────────────
    bool ok = sendToCloud(temperature, humidity, ldrValue,
                          soilValue, distance, servoAngle,
                          relay1, relay2, relay3, relay4);

    Serial.printf("[HTTP] %s\n", ok ? "Sent OK" : "FAILED — retry next cycle");
    Serial.println("──────────────────────────────────────────────");
  }
}

// ═══════════════════════════════════════════════
//  Relay control — LOW = ON, HIGH = OFF
// ═══════════════════════════════════════════════
void handleRelays(int r1, int r2, int r3, int r4) {
  digitalWrite(RELAY1_PIN, r1 ? LOW : HIGH);
  digitalWrite(RELAY2_PIN, r2 ? LOW : HIGH);
  digitalWrite(RELAY3_PIN, r3 ? LOW : HIGH);
  digitalWrite(RELAY4_PIN, r4 ? LOW : HIGH);
}

// ═══════════════════════════════════════════════
//  Feeder timer — independent of data sending
//  ON for 5 seconds every 1 minute
// ═══════════════════════════════════════════════
void handleFeeder() {
  unsigned long now = millis();

  if (!feederRunning) {
    // ── Wait for 1 minute interval ─────────────
    if (now - lastFeederTime >= FEEDER_INTERVAL || lastFeederTime == 0) {
      feederRunning   = true;
      feederStartTime = now;
      lastFeederTime  = now;
      digitalWrite(RELAY2_PIN, LOW);   // Relay2 ON
      Serial.println("[FEEDER] Started — sweeping 0° to 90° over 5 seconds");
    }

  } else {
    // ── Feeder is running — sweep servo 0→90 ───
    unsigned long elapsed = now - feederStartTime;

    if (elapsed < FEEDER_ON_TIME) {
      // Smoothly map elapsed time (0→5000ms) to angle (0°→90°)
      int angle = map(elapsed, 0, FEEDER_ON_TIME, 0, 90);
      myServo.write(angle);

      // Print angle every ~500ms to avoid flooding Serial
      static unsigned long lastPrint = 0;
      if (now - lastPrint >= 500) {
        Serial.printf("[FEEDER] Servo angle: %d°\n", angle);
        lastPrint = now;
      }

    } else {
      // ── 5 seconds done — return to 0° ────────
      feederRunning = false;
      myServo.write(0);                 // back to 0°
      digitalWrite(RELAY2_PIN, HIGH);  // Relay2 OFF
      Serial.println("[FEEDER] Done — servo back to 0° — next feed in 1 min");
    }
  }
}
// ═══════════════════════════════════════════════
//  Smoothed analog read — reduces noise
// ═══════════════════════════════════════════════
int smoothAnalog(int pin, int samples) {
  long sum = 0;
  for (int i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delay(3);
  }
  return (int)(sum / samples);
}

// ═══════════════════════════════════════════════
//  Ultrasonic HC-SR04
// ═══════════════════════════════════════════════
float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duration == 0) return -1.0;
  return duration * 0.0343 / 2.0;
}

// ═══════════════════════════════════════════════
//  Send all data to Render backend via HTTP GET
// ═══════════════════════════════════════════════
bool sendToCloud(float temp, float hum, int ldr,
                 int soil, float dist, int servo_angle,
                 int r1, int r2, int r3, int r4) {

  if (WiFi.status() != WL_CONNECTED) return false;

  String url = String(SERVER_URL)
    + "?temperature="   + String(temp, 2)
    + "&humidity="      + String(hum,  2)
    + "&ldr_value="     + String(ldr)
    + "&soil_moisture=" + String(soil)
    + "&distance="      + String(dist, 2)
    + "&servo_angle="   + String(servo_angle)
    + "&relay1="        + String(r1)
    + "&relay2="        + String(r2)
    + "&relay3="        + String(r3)
    + "&relay4="        + String(r4);

  HTTPClient http;
  http.begin(url);
  http.setTimeout(10000);

  int code = http.GET();

  if (code > 0) {
    String resp = http.getString();
    Serial.printf("[HTTP] %d → %s\n", code, resp.c_str());
    http.end();
    return true;
  } else {
    Serial.printf("[HTTP] Error: %s\n", http.errorToString(code).c_str());
    http.end();
    return false;
  }
}

// ═══════════════════════════════════════════════
//  WiFi connect with retry loop
// ═══════════════════════════════════════════════
void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (++attempts > 40) {
      Serial.println("\n[WiFi] Timeout — retrying...");
      WiFi.disconnect();
      delay(1000);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      attempts = 0;
    }
  }
  Serial.printf("\n[WiFi] Connected — IP: %s\n",
                WiFi.localIP().toString().c_str());
}
