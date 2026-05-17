-- ============================================================
-- SmartFarm IoT — Database Schema (MySQL Version)
-- ============================================================

CREATE DATABASE IF NOT EXISTS esp32_db;

USE esp32_db;

-- ── Main sensor readings table ────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_data (

    id INT PRIMARY KEY AUTO_INCREMENT,

    -- Auto-filled by DB on insert
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- DHT22
    temperature FLOAT NOT NULL,

    humidity FLOAT NOT NULL,

    -- LDR
    ldr_value INT NOT NULL,

    -- Relay status
    light_status INT NOT NULL DEFAULT 0
);

