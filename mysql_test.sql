-- ============================================================
-- SmartFarm IoT — MySQL Database Schema
-- Light + Climate Monitoring
-- ============================================================

-- Create database
CREATE DATABASE IF NOT EXISTS smartfarm_db;

-- Select database
USE smartfarm_db;

-- ============================================================
-- Main Sensor Readings Table
-- ============================================================

CREATE TABLE IF NOT EXISTS sensor_data (
    
    -- Unique ID
    id INT AUTO_INCREMENT PRIMARY KEY,

    -- Auto timestamp
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- DHT22 values
    temperature FLOAT NOT NULL,      -- Celsius
    humidity FLOAT NOT NULL,         -- Percentage

    -- LDR value from ESP32 ADC
    ldr_value INT NOT NULL,          -- 0 - 4095

    -- Relay / Light status
    light_status TINYINT(1) NOT NULL DEFAULT 0

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- Index for Faster Queries
-- ============================================================

CREATE INDEX idx_timestamp
ON sensor_data(timestamp);

-- ============================================================
-- Sample Test Data
-- ============================================================

INSERT INTO sensor_data
(temperature, humidity, ldr_value, light_status)
VALUES
(28.5, 64.2, 1200, 0);
