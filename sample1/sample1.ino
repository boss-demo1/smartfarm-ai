/**
 * SmartFarm IoT — ESP32 Firmware (Light & Climate Only)
 * ─────────────────────────────────────────────────────────────
 * Sensors : DHT22, LDR
 * Actuators: Relay (lights)
 * Protocol : HTTP POST → Flask backend (JSON payload)
 * Interval : 60 seconds
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ─── WiFi Credentials ─────────────────────────────────────────
const char* WIFI_SSID     = "11";
const char* WIFI_PASSWORD = "1234567890";

// ─── Backend Server ───────────────────────────────────────────
const char* SERVER_URL    = "http://10.131.61.21:5000/sensor-data";

// ─── Pin Definitions ──────────────────────────────────────────
#define PIN_LDR          34   // Analog — LDR voltage divider
#define PIN_DHT          4    // Digital — DHT22 data
#define PIN_RELAY_LIGHT  14   // Digital — Relay (grow light)

// ─── DHT Setup ────────────────────────────────────────────────
#define DHT_TYPE DHT22
DHT dht(PIN_DHT, DHT_TYPE);

// ─── Timing ───────────────────────────────────────────────────
const unsigned long SEND_INTERVAL_MS = 60000UL;   // 60 seconds
unsigned long lastSendTime = 0;

// ─── Automation Thresholds ────────────────────────────────────
const int  LDR_DARK_THRESHOLD  = 300;   // < 300 → lights ON

// ─── Prototypes ───────────────────────────────────────────────
void connectWiFi();
void applyAutomation(int ldr);
bool sendData(float temp, float hum, int ldr);

// ═════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n=== SmartFarm IoT Booting (Reduced Sensors) ===");

  // Light Relay — start OFF (HIGH = relay off for most active-low boards)
  pinMode(PIN_RELAY_LIGHT, OUTPUT);
  digitalWrite(PIN_RELAY_LIGHT, HIGH);

  // DHT
  dht.begin();

  // WiFi
  connectWiFi();

  Serial.println("=== Boot complete. Sending first reading… ===\n");
}

// ═════════════════════════════════════════════════════════════
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected — reconnecting…");
    connectWiFi();
  }

  unsigned long now = millis();
  if (now - lastSendTime >= SEND_INTERVAL_MS || lastSendTime == 0) {
    lastSendTime = now;

    // ── Read Sensors ──────────────────────────────────────────
    float temperature = dht.readTemperature();
    float humidity    = dht.readHumidity();

    if (isnan(temperature) || isnan(humidity)) {
      Serial.println("[DHT22] Read failed — skipping cycle");
      return;
    }

    int ldrValue = analogRead(PIN_LDR);

    // ── Print to Serial Monitor ───────────────────────────────
    Serial.println("─────────────────────────────────");
    Serial.printf("[Sensor] Temp      : %.2f °C\n",   temperature);
    Serial.printf("[Sensor] Humidity  : %.2f %%\n",   humidity);
    Serial.printf("[Sensor] LDR       : %d\n",        ldrValue);

    // ── Automation ────────────────────────────────────────────
    applyAutomation(ldrValue);

    // ── Send to Server ────────────────────────────────────────
    bool ok = sendData(temperature, humidity, ldrValue);
    if (!ok) {
      Serial.println("[HTTP] Will retry next cycle.");
    }
    Serial.println("─────────────────────────────────\n");
  }
}

// ─────────────────────────────────────────────────────────────
void applyAutomation(int ldr) {
  if (ldr < LDR_DARK_THRESHOLD) {
    digitalWrite(PIN_RELAY_LIGHT, LOW); // Relay ON
    Serial.println("[Auto] Dark env  → Light ON");
  } else {
    digitalWrite(PIN_RELAY_LIGHT, HIGH); // Relay OFF
    Serial.println("[Auto] Bright env→ Light OFF");
  }
}

// ─────────────────────────────────────────────────────────────
bool sendData(float temp, float hum, int ldr) {
  if (WiFi.status() != WL_CONNECTED) return false;

  StaticJsonDocument<128> doc;
  doc["temperature"]    = serialized(String(temp, 2));
  doc["humidity"]       = serialized(String(hum, 2));
  doc["ldr_value"]      = ldr;

  String jsonBody;
  serializeJson(doc, jsonBody);

  Serial.printf("[HTTP] POST → %s\n", SERVER_URL);
  Serial.printf("[HTTP] Body: %s\n",  jsonBody.c_str());

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  int code = http.POST(jsonBody);

  if (code > 0) {
    String resp = http.getString();
    Serial.printf("[HTTP] Response %d : %s\n", code, resp.c_str());
    http.end();
    return (code == 201 || code == 200);
  } else {
    Serial.printf("[HTTP] Error: %s\n", http.errorToString(code).c_str());
    http.end();
    return false;
  }
}

// ─────────────────────────────────────────────────────────────
void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (++attempts > 40) {
      Serial.println("\n[WiFi] Timeout — restarting WiFi");
      WiFi.disconnect();
      delay(1000);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      attempts = 0;
    }
  }
  Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
}