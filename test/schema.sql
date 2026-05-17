-- SmartFarm IoT — MySQL Schema
-- Run this once in Railway's query console after creating the database

CREATE TABLE IF NOT EXISTS sensor_data (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    timestamp    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    temperature  FLOAT        NOT NULL,
    humidity     FLOAT        NOT NULL,
    ldr_value    INT          NOT NULL,
    light_status TINYINT(1)   NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_data (timestamp);

-- Verify
SELECT COUNT(*) AS total_rows FROM sensor_data;
