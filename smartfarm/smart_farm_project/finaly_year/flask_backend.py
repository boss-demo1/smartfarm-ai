
from flask import Flask, request, jsonify
from datetime import date, datetime, timedelta
import os
from models import db, SensorReading, FarmConfig
from predictors import predict_irrigation, predict_yield

# APP & DATABASE SETUP

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(BASE_DIR, 'farm.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()
    if not FarmConfig.query.first():
        db.session.add(FarmConfig(sowing_date=date.today()))
        db.session.commit()
        print(f"[INIT] Database created. Sowing date set to: {date.today()}")

GDD_BASE    = 10.0
GDD_UPPER   = 30.0
TEMP_OPT_LO = 18.0
TEMP_OPT_HI = 29.0


def compute_cumulatives(new_reading: SensorReading) -> dict:
    config     = FarmConfig.query.first()
    sowing     = config.sowing_date

    # Temporary fix for testing: Every 20 sensor pings counts as 1 "day" passing
    all_rows = SensorReading.query.count()
    day_number = max(1, all_rows )

    # Fetch ALL rows including the one just committed
    all_rows = SensorReading.query.order_by(SensorReading.timestamp).all()

    def gdd(t):
        """Daily GDD contribution from temperature t."""
        return max(0.0, min(t, GDD_UPPER) - GDD_BASE)

    def temp_stress(t):
        """Degrees outside the optimal temperature band."""
        return max(0.0, TEMP_OPT_LO - t) + max(0.0, t - TEMP_OPT_HI)

    cum_gdd             = sum(gdd(r.temp)        for r in all_rows)
    cum_water           = sum(r.pump_seconds      for r in all_rows)
    cum_temp_stress     = sum(temp_stress(r.temp) for r in all_rows)
    cum_humidity_stress = sum(1 for r in all_rows if r.humidity > 80)
    cum_light           = sum(r.light             for r in all_rows)

    return {
        "day_number":               day_number,
        "cumulative_gdd":           round(cum_gdd, 2),
        "cumulative_water":         round(cum_water, 2),
        "cumulative_temp_stress":   round(cum_temp_stress, 2),
        "cumulative_humidity_stress": cum_humidity_stress,
        "cumulative_light":         cum_light,
    }


# ─────────────────────────────────────────────────────────────
# ROUTE 1 — POST /sensor  (called by ESP32 every 30 s)
# ─────────────────────────────────────────────────────────────

@app.route("/sensor", methods=["POST"])
def sensor_endpoint():

    #  Parse request
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    required = ["soil", "temp", "humidity", "light", "water_level", "time_day"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    soil        = float(data["soil"])
    temp        = float(data["temp"])
    humidity    = float(data["humidity"])
    light       = float(data["light"])
    water_level = float(data["water_level"])
    time_day    = int(data["time_day"])

    # ── Step 1: Irrigation prediction ────────────────────────
    pump_seconds = predict_irrigation(
        soil=soil, temp=temp, humidity=humidity,
        light=light, water_level=water_level, time_day=time_day,
        model_dir=MODEL_DIR
    )

    # ── Step 2: Persist sensor reading + pump decision ────────
    reading = SensorReading(
        soil=soil, temp=temp, humidity=humidity,
        light=light, water_level=water_level,
        time_day=time_day, pump_seconds=pump_seconds
    )
    db.session.add(reading)
    db.session.commit()   # commit before computing cumulatives

    # ── Step 3: Cumulative features from full history ─────────
    cums = compute_cumulatives(reading)

    # ── Step 4: Yield prediction ──────────────────────────────
    yield_result = predict_yield(
        cumulative_gdd=cums["cumulative_gdd"],
        cumulative_water=cums["cumulative_water"],
        cumulative_temp_stress=cums["cumulative_temp_stress"],
        cumulative_humidity_stress=cums["cumulative_humidity_stress"],
        cumulative_light=cums["cumulative_light"],
        day_number=cums["day_number"],
        temp=temp, humidity=humidity, soil=soil,
        light=light, water_level=water_level,
        model_dir=MODEL_DIR
    )

    # ── Step 5: Build and return response to ESP32 ────────────
    response = {
        "status":     "ok",
        "irrigation": {"pump_seconds": pump_seconds},   # ← ESP32 reads this key
        "yield":      yield_result,
        "farm_state": cums,
    }

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"T={temp}°C  H={humidity}%  Soil={soil}  "
        f"→ Pump={pump_seconds}s | "
        f"Yield={yield_result['grams']}g | "
        f"GDD={cums['cumulative_gdd']}  Day={cums['day_number']}"
    )

    return jsonify(response), 200


# ─────────────────────────────────────────────────────────────
# ROUTE 2 — GET /status
# ─────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """Return the latest sensor reading and current cumulative farm state."""

    last   = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
    config = FarmConfig.query.first()
    sowing = config.sowing_date

    all_r      = SensorReading.query.order_by(SensorReading.timestamp).all()
    day_number = max(1, (date.today() - sowing).days + 1)

    def gdd(t):
        return max(0.0, min(t, GDD_UPPER) - GDD_BASE)

    def ts(t):
        return max(0.0, TEMP_OPT_LO - t) + max(0.0, t - TEMP_OPT_HI)

    cums = {
        "day_number":               day_number,
        "cumulative_gdd":           round(sum(gdd(r.temp) for r in all_r), 2),
        "cumulative_water":         round(sum(r.pump_seconds for r in all_r), 2),
        "cumulative_temp_stress":   round(sum(ts(r.temp) for r in all_r), 2),
        "cumulative_humidity_stress": sum(1 for r in all_r if r.humidity > 80),
        "cumulative_light":         sum(r.light for r in all_r),
        "total_readings":           len(all_r),
        "sowing_date":              str(sowing),
    }

    return jsonify({
        "latest_reading": last.to_dict() if last else None,
        "farm_state":     cums,
    }), 200


# ─────────────────────────────────────────────────────────────
# ROUTE 3 — GET /history
# ─────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
def history():
    """
    Return sensor readings from the last N days.
    Usage: GET /history?days=7
    """
    days  = int(request.args.get("days", 7))
    since = datetime.utcnow() - timedelta(days=days)

    rows = (
        SensorReading.query
        .filter(SensorReading.timestamp >= since)
        .order_by(SensorReading.timestamp.desc())
        .limit(500)
        .all()
    )

    return jsonify([r.to_dict() for r in rows]), 200


# ─────────────────────────────────────────────────────────────
# ROUTE 4 — POST /reset_farm
# ─────────────────────────────────────────────────────────────

@app.route("/reset_farm", methods=["POST"])
def reset_farm():
    """
    Reset the sowing date when you start a new crop cycle.
    Optionally pass: { "sowing_date": "2026-06-10" }
    If omitted, today's date is used.
    """
    data       = request.get_json(force=True, silent=True) or {}
    sowing_str = data.get("sowing_date")
    new_sowing = (
        date.fromisoformat(sowing_str) if sowing_str else date.today()
    )

    config             = FarmConfig.query.first()
    config.sowing_date = new_sowing
    db.session.commit()

    return jsonify({
        "status":          "reset",
        "new_sowing_date": str(new_sowing),
    }), 200

# ENTRY POINT
if __name__ == "__main__":
    # host="0.0.0.0" makes the server reachable from the ESP32 on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)
