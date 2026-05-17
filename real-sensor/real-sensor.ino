#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"

#define DHTPIN 4
#define DHTTYPE DHT22

#define LDR_PIN 34
#define RELAY_PIN 14

const char* ssid = "11";
const char* password = "1234567890";

/*
IMPORTANT:
Replace with your Linux Mint IP address

Example:
http://192.168.1.5:5000/upload
*/
const char* serverName = "http://10.69.212.21:5000/upload";

DHT dht(DHTPIN, DHTTYPE);

void setup() {

  Serial.begin(115200);

  pinMode(RELAY_PIN, OUTPUT);

  dht.begin();

  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected");
  Serial.println(WiFi.localIP());
}

void loop() {

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  int ldrValue = analogRead(LDR_PIN);

  int lightStatus;

  // Relay Logic
  if (ldrValue < 1500) {
    digitalWrite(RELAY_PIN, HIGH);
    lightStatus = 0;
  }
  else {
    digitalWrite(RELAY_PIN, LOW);
    lightStatus = 1;
  }

  // Check sensor validity
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read DHT sensor");
    delay(2000);
    return;
  }

  Serial.println("Sending Data...");

  Serial.print("Temperature: ");
  Serial.println(temperature);

  Serial.print("Humidity: ");
  Serial.println(humidity);

  Serial.print("LDR: ");
  Serial.println(ldrValue);

  if (WiFi.status() == WL_CONNECTED) {

    HTTPClient http;

    String url = String(serverName)
      + "?temperature=" + String(temperature)
      + "&humidity=" + String(humidity)
      + "&ldr_value=" + String(ldrValue)
      + "&light_status=" + String(lightStatus);

    http.begin(url);

    int httpResponseCode = http.GET();

    Serial.print("HTTP Response: ");
    Serial.println(httpResponseCode);

    http.end();
  }

  delay(5000);
}
