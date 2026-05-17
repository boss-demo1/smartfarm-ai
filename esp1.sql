CREATE TABLE sensor_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    temperature FLOAT NOT NULL,
    humidity FLOAT NOT NULL,
    ldr_value INT NOT NULL,
    light_status TINYINT(1) NOT NULL DEFAULT 0
);
---create the index
CREATE INDEX idx_timestamp
ON sensor_data(timestamp);
