"""
models.py
---------
SQLAlchemy database models.

Tables created automatically by flaa.py on first run:
  - sensor_reading  : every sensor snapshot + pump decision
  - farm_config     : one row holding the sowing date
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Shared db instance — imported by flaa.py
db = SQLAlchemy()


# ─────────────────────────────────────────────────────────────
# SensorReading
# ─────────────────────────────────────────────────────────────

class SensorReading(db.Model):
    """
    One row per POST /sensor call.
    Stores the raw sensor values AND the pump decision from the ML model.
    """

    __tablename__ = "sensor_reading"

    # Primary key — auto-incremented integer
    id = db.Column(db.Integer, primary_key=True)

    # When this row was inserted (UTC)
    timestamp = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # ── Sensor values ────────────────────────────────────────
    soil        = db.Column(db.Float, nullable=False)   # ADC 0-4095
    temp        = db.Column(db.Float, nullable=False)   # °C
    humidity    = db.Column(db.Float, nullable=False)   # %
    light       = db.Column(db.Float, nullable=False)   # ADC 0-1023
    water_level = db.Column(db.Float, nullable=False)   # % (0-100)
    time_day    = db.Column(db.Integer, nullable=False) # hour 0-23

    # ── ML output ────────────────────────────────────────────
    pump_seconds = db.Column(db.Float, nullable=False)  # seconds to run pump

    def to_dict(self):
        """Convert row to JSON-serialisable dict."""
        return {
            "id":           self.id,
            "timestamp":    self.timestamp.isoformat(),
            "soil":         self.soil,
            "temp":         self.temp,
            "humidity":     self.humidity,
            "light":        self.light,
            "water_level":  self.water_level,
            "time_day":     self.time_day,
            "pump_seconds": self.pump_seconds,
        }


# ─────────────────────────────────────────────────────────────
# FarmConfig
# ─────────────────────────────────────────────────────────────

class FarmConfig(db.Model):
    """
    Single-row config table.
    Stores the sowing date so cumulative GDD can be calculated
    relative to planting day.
    Reset this via POST /reset_farm when you start a new crop cycle.
    """

    __tablename__ = "farm_config"

    id          = db.Column(db.Integer, primary_key=True)

    # Date seeds were planted — used to compute day_number
    sowing_date = db.Column(db.Date, nullable=False)
