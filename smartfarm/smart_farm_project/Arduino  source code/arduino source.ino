#define BLYNK_TEMPLATE_ID "TMPL2n6YM8hBM"
#define BLYNK_TEMPLATE_NAME "AQUA"
#define BLYNK_AUTH_TOKEN "8C_b0q95ICCtzZZo4pnpYMeQL8A0R8_j"        // <-- Replace

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <DHT.h>
#include <ESP32Servo.h>
#include <BlynkSimpleEsp32.h>

// =====================================================
// DHT22
// =====================================================

#define DHTPIN   4
#define DHTTYPE  DHT22

DHT dht(DHTPIN, DHTTYPE);

// =====================================================
// SENSOR PINS
// =====================================================

#define SOIL_PIN  35
#define LDR_PIN   34
#define TRIG_PIN  27
#define ECHO_PIN  33

// =====================================================
// OUTPUT PINS
// =====================================================

#define SERVO_PIN    25
#define PUMP1_ML     14   // Irrigation pump (ML controlled)
#define PUMP2_TANK   13   // Tank refill pump

// =====================================================
// BLYNK VIRTUAL PINS
// =====================================================
//  V0  = Soil Moisture   (Gauge)
//  V2  = Fish Feed       (Button - momentary)
//  V3  = Terminal        (ML output log)
//  V4  = Temperature     (Gauge)
//  V5  = Humidity        (Gauge)
//  V7  = Water Level     (Gauge)
//  V10 = Pump1 status    (LED)
//  V11 = Pump2 status    (LED)
//  V16 = pump1 manual    (Button)
//  V17 = pump2 manual    (Button)

WidgetTerminal terminal(V18);
WidgetLED      pump1LED(V10);
WidgetLED      pump2LED(V11);

// =====================================================
// TANK SETTINGS
// =====================================================

const float TANK_EMPTY_DISTANCE = 50.0;
const float TANK_FULL_DISTANCE  = 10.0;

// =====================================================
// WIFI & SERVER
// =====================================================

const char* WIFI_SSID     = "Galaxy";
const char* WIFI_PASSWORD = "22111122";

const char* SERVER_URL = "http://192.168.81.177:5000/sensor";

// =====================================================
// NTP TIME
// =====================================================

WiFiUDP   ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 3600);

// =====================================================
// SERVO
// =====================================================

Servo myServo;

unsigned long lastServoActionTime = 0;
const unsigned long SERVO_INTERVAL = 60000;   // feed every 1 min

// =====================================================
// ML PUMP (PUMP1) VARIABLES
// =====================================================

bool          pump_running       = false;
unsigned long pump_start         = 0;
unsigned long pump_duration_ms   = 0;

// =====================================================
// TANK PUMP (PUMP2) VARIABLES
// =====================================================

bool          pump2IsRunning     = false;
unsigned long pump2_start_time   = 0;
const unsigned long PUMP2_DURATION_MS = 7000;   // run for exactly 7 sec

// =====================================================
// SENSOR SEND TIMER
// =====================================================

const unsigned long SEND_INTERVAL_MS = 30000;
unsigned long last_send_time = 0;

// =====================================================
// BLYNK: FISH FEED BUTTON (V2)
// =====================================================

BLYNK_WRITE(V2) {
  if (param.asInt() == 1) {
    Serial.println("[Blynk] Manual Fish Feed");
    terminal.println("[Feed] Manual triggered");
    terminal.flush();

    myServo.write(180);
    delay(5000);
    myServo.write(0);

    lastServoActionTime = millis(); // reset auto-feed timer
  }
}

// =====================================================
// BLYNK: PUMP1 MANUAL CONTROL (V16)
// =====================================================

BLYNK_WRITE(V16) {
  if (param.asInt() == 1) {
    Serial.println("[Blynk] Pump1 Manual ON");
    terminal.println("[Pump1] Manual ON");
    terminal.flush();
    digitalWrite(PUMP1_ML, LOW);
    pump1LED.on();
  } else {
    Serial.println("[Blynk] Pump1 Manual OFF");
    terminal.println("[Pump1] Manual OFF");
    terminal.flush();
    digitalWrite(PUMP1_ML, HIGH);
    pump_running = false;
    pump1LED.off();
  }
}

// =====================================================
// BLYNK: PUMP2 MANUAL CONTROL (V17)
// =====================================================

BLYNK_WRITE(V17) {
  if (param.asInt() == 1) {
    Serial.println("[Blynk] Pump2 Manual ON");
    terminal.println("[Pump2] Manual ON");
    terminal.flush();
    digitalWrite(PUMP2_TANK, LOW);
    pump2IsRunning    = true;
    pump2_start_time  = millis();
    pump2LED.on();
  } else {
    Serial.println("[Blynk] Pump2 Manual OFF");
    terminal.println("[Pump2] Manual OFF");
    terminal.flush();
    digitalWrite(PUMP2_TANK, HIGH);
    pump2IsRunning = false;
    pump2LED.off();
  }
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);

  dht.begin();

  // --- Servo ---
  myServo.attach(SERVO_PIN);
  myServo.write(0);

  // --- Ultrasonic ---
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // --- Relays (HIGH = OFF for active-low relay) ---
  pinMode(PUMP1_ML,   OUTPUT);
  pinMode(PUMP2_TANK, OUTPUT);
  digitalWrite(PUMP1_ML,   HIGH);
  digitalWrite(PUMP2_TANK, HIGH);

  // --- WiFi ---
  Serial.print("Connecting WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi Connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  // --- Blynk ---
  Blynk.config(BLYNK_AUTH_TOKEN);
  Blynk.connect();

  // --- NTP ---
  timeClient.begin();
}

// =====================================================
// ULTRASONIC DISTANCE
// =====================================================

float getRawDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return TANK_EMPTY_DISTANCE;

  return duration * 0.034 / 2.0;
}

// =====================================================
// WATER PERCENTAGE
// =====================================================

float getWaterLevelPercentage() {
  float distance = getRawDistance();

  if (distance < TANK_FULL_DISTANCE)  distance = TANK_FULL_DISTANCE;
  if (distance > TANK_EMPTY_DISTANCE) distance = TANK_EMPTY_DISTANCE;

  return ((TANK_EMPTY_DISTANCE - distance) /
          (TANK_EMPTY_DISTANCE - TANK_FULL_DISTANCE)) * 100.0;
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop() {

  Blynk.run();

  unsigned long now = millis();

  // --- Update internet time ---
  if (WiFi.status() == WL_CONNECTED) {
    timeClient.update();
  }

  // -----------------------------------------------
  // STOP ML PUMP (PUMP1) when duration expires
  // -----------------------------------------------
  if (pump_running && (now - pump_start >= pump_duration_ms)) {
    digitalWrite(PUMP1_ML, HIGH);
    pump_running = false;
    pump1LED.off();
    Serial.println("[PUMP1] OFF - duration done");
    terminal.println("[Pump1] OFF - irrigation complete");
    terminal.flush();
  }

  // -----------------------------------------------
  // STOP TANK PUMP (PUMP2) after exactly 7 seconds
  // -----------------------------------------------
  if (pump2IsRunning && (now - pump2_start_time >= PUMP2_DURATION_MS)) {
    digitalWrite(PUMP2_TANK, HIGH);
    pump2IsRunning = false;
    pump2LED.off();
    Serial.println("[PUMP2] OFF - 7 sec done");
    terminal.println("[Pump2] OFF - 7 sec refill done");
    terminal.flush();
  }

  // -----------------------------------------------
  // FISH FEEDER (auto, every SERVO_INTERVAL)
  // -----------------------------------------------
  if (now - lastServoActionTime >= SERVO_INTERVAL) {
    Serial.println("Auto Feeding Fish");
    terminal.println("[Feed] Auto feeding fish");
    terminal.flush();

    myServo.write(180);
    delay(5000);
    myServo.write(0);

    lastServoActionTime = now;
  }

  // -----------------------------------------------
  // SEND SENSOR DATA every SEND_INTERVAL_MS
  // -----------------------------------------------
  if (now - last_send_time >= SEND_INTERVAL_MS) {
    last_send_time = now;
    read_and_send();
  }
}

// =====================================================
// READ SENSORS AND SEND TO FLASK
// =====================================================

void read_and_send() {

  // --- Read sensors ---
  int   soil             = analogRead(SOIL_PIN);
  int   light            = map(analogRead(LDR_PIN), 0, 4095, 0, 1023);
  float temp             = dht.readTemperature();
  float humidity         = dht.readHumidity();
  float water_percentage = getWaterLevelPercentage();
  int   time_day         = timeClient.getHours();

  // --- Validate DHT ---
  if (isnan(temp) || isnan(humidity)) {
    Serial.println("[ERROR] DHT read failed");
    terminal.println("[ERROR] DHT read failed");
    terminal.flush();
    return;
  }

  // -----------------------------------------------
  // TANK REFILL LOGIC
  // Start pump if low & not already running.
  // The loop() timer will cut it off at 7 seconds.
  // -----------------------------------------------
  if (water_percentage < 30.0 && !pump2IsRunning) {
    Serial.println("Tank low -> Pump2 ON (7 sec)");
    terminal.println("[Pump2] Tank low - ON for 7 sec");
    terminal.flush();
    digitalWrite(PUMP2_TANK, LOW);
    pump2IsRunning   = true;
    pump2_start_time = millis();
    pump2LED.on();
  }

  // --- Serial monitor ---
  Serial.println("=========================");
  Serial.print("Soil: ");      Serial.println(soil);
  Serial.print("Light: ");     Serial.println(light);
  Serial.print("Temp: ");      Serial.println(temp);
  Serial.print("Humidity: ");  Serial.println(humidity);
  Serial.print("Water %: ");   Serial.println(water_percentage);
  Serial.print("Hour: ");      Serial.println(time_day);

  // --- Push to Blynk gauges ---
  Blynk.virtualWrite(V0, soil);
  Blynk.virtualWrite(V4, temp);
  Blynk.virtualWrite(V5, humidity);
  Blynk.virtualWrite(V7, water_percentage);

  // --- Build JSON ---
  StaticJsonDocument<2048> doc;
  doc["soil"]        = soil;
  doc["temp"]        = temp;
  doc["humidity"]    = humidity;
  doc["light"]       = light;
  doc["water_level"] = water_percentage;
  doc["time_day"]    = time_day;

  String payload;
  serializeJson(doc, payload);

  // --- Check WiFi ---
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected");
    terminal.println("[WiFi] Disconnected - skipping send");
    terminal.flush();
    return;
  }

  // --- POST to Flask ---
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int http_code = http.POST(payload);

  // -----------------------------------------------
  // HANDLE FLASK RESPONSE
  // -----------------------------------------------
  if (http_code == 200) {

    String response = http.getString();
    Serial.println(response);

    StaticJsonDocument<512> resp;
    DeserializationError err = deserializeJson(resp, response);

    if (!err) {

      // --- Parse ML fields ---
      const char* growth_stage  = resp["growth_stage"]  | "N/A";
      const char* harvest_date  = resp["harvest_date"]  | "N/A";
      int   days_remaining      = resp["days_remaining"] | 0;
      float grams               = resp["grams"]          | 0.0;
      float pods                = resp["pods"]            | 0.0;
      float pump_seconds        = resp["irrigation"]["pump_seconds"] | 0.0;

      // --- Send ML info to Blynk terminal ---
      terminal.println("=== ML RESPONSE ===");
      terminal.print("Stage: ");      terminal.println(growth_stage);
      terminal.print("Harvest: ");    terminal.println(harvest_date);
      terminal.print("Days Left: ");  terminal.println(days_remaining);
      terminal.print("Grams: ");      terminal.println(grams);
      terminal.print("Pods: ");       terminal.println(pods);
      terminal.print("Pump Sec (raw): "); terminal.println(pump_seconds);

      Serial.print("Pump Seconds (raw): ");
      Serial.println(pump_seconds);

      // --- Start ML pump with +2 sec buffer ---
      if (pump_seconds > 0) {
        float total_seconds  = pump_seconds + 2.0;   // +2 sec added here
        pump_duration_ms     = (unsigned long)(total_seconds * 1000);
        pump_start           = millis();
        pump_running         = true;

        digitalWrite(PUMP1_ML, LOW);
        pump1LED.on();

        Serial.print("[PUMP1] ON for ");
        Serial.print(total_seconds);
        Serial.println(" sec");

        terminal.print("Pump ON for: ");
        terminal.print(total_seconds);
        terminal.println(" sec (+2 buffer)");
      } else {
        terminal.println("No irrigation needed");
      }

      terminal.println("===================");
      terminal.flush();

    } else {
      Serial.println("[ERROR] JSON Parse Failed");
      terminal.println("[ERROR] JSON parse failed");
      terminal.flush();
    }

  } else {
    Serial.print("[HTTP ERROR] ");
    Serial.println(http_code);
    terminal.print("[HTTP ERROR] ");
    terminal.println(http_code);
    terminal.flush();
  }

  http.end();
}
