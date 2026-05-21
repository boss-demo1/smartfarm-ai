"""
SmartFarm IoT v2.0 — Cloud Backend
Flask + MySQL (Railway) + Auto-retraining ML
Sensors  : DHT22, LDR, Soil Moisture, Ultrasonic
Relays   : Pump1, Feeder, Pump2, Light
"""

import os
import io
import csv
import logging
import threading
from datetime import datetime

import pandas as pd
import joblib
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error as MySQLError
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Constants ─────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "smartfarm_model.pkl")
RETRAIN_EVERY = 50

# ── Automation thresholds — must match ESP32 sketch ───────────
ULTRASONIC_LOW_CM  = 10.0   # below 10cm  → Pump1 ON
SOIL_DRY_THRESHOLD = 2500   # above 2500  → Pump2 ON
LDR_DARK_THRESHOLD = 1500   # below 1500  → Light ON

# ── Model state ───────────────────────────────────────────────
model_state = {
    "model":          None,
    "trained_on":     0,
    "last_retrained": None,
    "lock":           threading.Lock()
}

try:
    model_state["model"] = joblib.load(MODEL_PATH)
    log.info("ML model loaded from %s", MODEL_PATH)
except FileNotFoundError:
    log.warning("No model found — rule-based fallback active")

# ── Database ──────────────────────────────────────────────────
def get_db():
    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env variables: {missing}")
    return mysql.connector.connect(
        host               = os.environ["DB_HOST"],
        user               = os.environ["DB_USER"],
        password           = os.environ["DB_PASSWORD"],
        database           = os.environ["DB_NAME"],
        port               = int(os.environ.get("DB_PORT", 3306)),
        connection_timeout = 10
    )

def init_db():
    """Create table with all sensor and relay columns."""
    sql = """
        CREATE TABLE IF NOT EXISTS sensor_data (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            timestamp     DATETIME   NOT NULL DEFAULT CURRENT_TIMESTAMP,
            temperature   FLOAT      NOT NULL,
            humidity      FLOAT      NOT NULL,
            ldr_value     INT        NOT NULL,
            soil_moisture INT        NOT NULL DEFAULT 0,
            distance      FLOAT      NOT NULL DEFAULT 0,
            relay1_pump1  TINYINT(1) NOT NULL DEFAULT 0,
            relay2_feeder TINYINT(1) NOT NULL DEFAULT 0,
            relay3_pump2  TINYINT(1) NOT NULL DEFAULT 0,
            relay4_light  TINYINT(1) NOT NULL DEFAULT 0
        )
    """
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        db.close()
        log.info("Database table ready")
    except Exception as e:
        log.error("Database init failed: %s", e)

# ── Rule-based automation ─────────────────────────────────────
def apply_rules(ldr: int, soil: int, distance: float) -> dict:
    """
    Derive relay states from sensor values.
    relay2 (feeder) is time-based — handled by ESP32 directly.
    Backend just records what ESP32 reported.
    """
    return {
        "relay1": 1 if (distance > 0 and distance < ULTRASONIC_LOW_CM) else 0,
        "relay2": 0,   # time-based — ESP32 controls this
        "relay3": 1 if soil > SOIL_DRY_THRESHOLD else 0,
        "relay4": 1 if ldr  < LDR_DARK_THRESHOLD else 0,
    }

# ── ML retrain ────────────────────────────────────────────────
def retrain_model():
    try:
        log.info("[RETRAIN] Starting background retrain...")
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT temperature, humidity, ldr_value,
                   soil_moisture, distance, relay4_light
            FROM sensor_data
        """)
        rows = cursor.fetchall()
        db.close()

        if len(rows) < 10:
            log.warning("[RETRAIN] Not enough rows (%d)", len(rows))
            return

        df = pd.DataFrame(rows)
        X  = df[["temperature", "humidity", "ldr_value",
                  "soil_moisture", "distance"]]
        y  = df["relay4_light"]  # predict light status

        new_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        new_model.fit(X, y)
        accuracy = round(new_model.score(X, y) * 100, 2)

        joblib.dump(new_model, MODEL_PATH)

        with model_state["lock"]:
            model_state["model"]          = new_model
            model_state["trained_on"]     = len(rows)
            model_state["last_retrained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log.info("[RETRAIN] Done — %d rows — accuracy %.2f%%", len(rows), accuracy)

    except Exception as e:
        log.error("[RETRAIN] Failed: %s", e)

def maybe_retrain():
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_data")
        total = cursor.fetchone()[0]
        db.close()

        if total - model_state["trained_on"] >= RETRAIN_EVERY:
            thread = threading.Thread(target=retrain_model, daemon=True)
            thread.start()
    except Exception as e:
        log.error("[RETRAIN CHECK] %s", e)

# ── ML prediction ─────────────────────────────────────────────
def predict_light(temperature, humidity, ldr, soil, distance):
    with model_state["lock"]:
        current_model = model_state["model"]

    if current_model is not None:
        df = pd.DataFrame([{
            "temperature":  temperature,
            "humidity":     humidity,
            "ldr_value":    ldr,
            "soil_moisture": soil,
            "distance":     distance
        }])
        return int(current_model.predict(df)[0])

    # Rule fallback
    return 1 if ldr < LDR_DARK_THRESHOLD else 0

def ts_fix(rows):
    """Convert datetime objects to strings for JSON."""
    for r in rows:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    return rows

# ── Error handler ─────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_error(e):
    log.error("Unhandled: %s", e)
    return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════

@app.route("/")
def home():
    with model_state["lock"]:
        has_model  = model_state["model"] is not None
        trained_on = model_state["trained_on"]
        last_rt    = model_state["last_retrained"]
    return jsonify({
        "status":         "SmartFarm API v2.0 online",
        "sensors":        ["DHT22", "LDR", "Soil Moisture", "Ultrasonic"],
        "relays":         ["Pump1(ultrasonic)", "Feeder(timer)",
                           "Pump2(soil)", "Light(ldr)"],
        "model":          "loaded" if has_model else "rule fallback",
        "trained_on":     f"{trained_on} rows",
        "last_retrained": last_rt or "never",
        "retrain_every":  f"{RETRAIN_EVERY} new rows"
    })

@app.route("/health")
def health():
    try:
        db = get_db()
        db.close()
        with model_state["lock"]:
            has_model = model_state["model"] is not None
        return jsonify({"db": "ok", "model": "loaded" if has_model else "fallback"}), 200
    except Exception as e:
        return jsonify({"db": "error", "detail": str(e)}), 500

@app.route("/model-status")
def model_status():
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_data")
        total = cursor.fetchone()[0]
        db.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    with model_state["lock"]:
        trained_on = model_state["trained_on"]
        last_rt    = model_state["last_retrained"]
        has_model  = model_state["model"] is not None

    return jsonify({
        "model_loaded":       has_model,
        "total_rows_in_db":   total,
        "trained_on_rows":    trained_on,
        "new_rows_since":     total - trained_on,
        "rows_until_retrain": RETRAIN_EVERY - ((total - trained_on) % RETRAIN_EVERY),
        "last_retrained":     last_rt or "never",
        "retrain_every":      RETRAIN_EVERY
    })

@app.route("/upload")
def upload():
    """
    GET /upload?temperature=&humidity=&ldr_value=
                &soil_moisture=&distance=
                &relay1=&relay2=&relay3=&relay4=

    ESP32 sends all sensor values + relay states it applied.
    Backend also runs ML prediction and saves everything.
    """
    try:
        temperature   = float(request.args.get("temperature",   0))
        humidity      = float(request.args.get("humidity",      0))
        ldr_value     = int(request.args.get("ldr_value",       0))
        soil_moisture = int(request.args.get("soil_moisture",   0))
        distance      = float(request.args.get("distance",      0))
        # Relay states reported by ESP32
        relay1        = int(request.args.get("relay1",          0))
        relay2        = int(request.args.get("relay2",          0))
        relay3        = int(request.args.get("relay3",          0))
        relay4        = int(request.args.get("relay4",          0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    # ML predicts light status independently for comparison
    ml_light = predict_light(temperature, humidity, ldr_value,
                             soil_moisture, distance)

    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO sensor_data
            (temperature, humidity, ldr_value, soil_moisture, distance,
             relay1_pump1, relay2_feeder, relay3_pump2, relay4_light)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (temperature, humidity, ldr_value, soil_moisture, distance,
              relay1, relay2, relay3, relay4))
        db.commit()
        new_id = cursor.lastrowid
        db.close()
    except MySQLError as e:
        return jsonify({"error": "DB write failed"}), 500

    maybe_retrain()

    log.info("id=%s temp=%.1f hum=%.1f ldr=%d soil=%d dist=%.1f "
             "r1=%d r2=%d r3=%d r4=%d",
             new_id, temperature, humidity, ldr_value,
             soil_moisture, distance, relay1, relay2, relay3, relay4)

    return jsonify({
        "status":             "saved",
        "id":                 new_id,
        "temperature":        temperature,
        "humidity":           humidity,
        "ldr_value":          ldr_value,
        "soil_moisture":      soil_moisture,
        "distance":           distance,
        "relay1_pump1":       relay1,
        "relay2_feeder":      relay2,
        "relay3_pump2":       relay3,
        "relay4_light":       relay4,
        "ml_light_prediction": ml_light
    })

@app.route("/predict")
def predict():
    """
    GET /predict?temperature=&humidity=&ldr_value=
                 &soil_moisture=&distance=
    Returns ML prediction + rule-based decisions for all relays.
    """
    try:
        temperature   = float(request.args.get("temperature",   0))
        humidity      = float(request.args.get("humidity",      0))
        ldr_value     = int(request.args.get("ldr_value",       0))
        soil_moisture = int(request.args.get("soil_moisture",   0))
        distance      = float(request.args.get("distance",      0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    rules    = apply_rules(ldr_value, soil_moisture, distance)
    ml_light = predict_light(temperature, humidity, ldr_value,
                             soil_moisture, distance)

    with model_state["lock"]:
        has_model  = model_state["model"] is not None
        trained_on = model_state["trained_on"]

    return jsonify({
        "inputs": {
            "temperature":   temperature,
            "humidity":      humidity,
            "ldr_value":     ldr_value,
            "soil_moisture": soil_moisture,
            "distance":      distance
        },
        "relay_decisions": {
            "relay1_pump1":  rules["relay1"],
            "relay2_feeder": "timer-based (ESP32)",
            "relay3_pump2":  rules["relay3"],
            "relay4_light":  ml_light
        },
        "reasons": {
            "relay1": f"distance {distance}cm {'< 10cm → ON' if rules['relay1'] else '>= 10cm → OFF'}",
            "relay2": "ON for 5s every 1 minute — controlled by ESP32 timer",
            "relay3": f"soil {soil_moisture} {'> 2500 → ON (dry)' if rules['relay3'] else '<= 2500 → OFF (wet)'}",
            "relay4": f"ldr {ldr_value} {'< 1500 → ON (dark)' if ml_light else '>= 1500 → OFF (bright)'}"
        },
        "model_used":       "ml" if has_model else "rule_fallback",
        "model_trained_on": f"{trained_on} rows"
    })

@app.route("/demo")
def demo():
    import random
    temperature   = round(random.uniform(20, 35), 2)
    humidity      = round(random.uniform(40, 80), 2)
    ldr_value     = random.randint(0, 4095)
    soil_moisture = random.randint(1000, 4000)
    distance      = round(random.uniform(2, 30), 2)
    rules         = apply_rules(ldr_value, soil_moisture, distance)
    ml_light      = predict_light(temperature, humidity, ldr_value,
                                  soil_moisture, distance)
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO sensor_data
            (temperature, humidity, ldr_value, soil_moisture, distance,
             relay1_pump1, relay2_feeder, relay3_pump2, relay4_light)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (temperature, humidity, ldr_value, soil_moisture, distance,
              rules["relay1"], 0, rules["relay3"], ml_light))
        db.commit()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    maybe_retrain()

    return jsonify({
        "status":        "demo row inserted",
        "temperature":   temperature,
        "humidity":      humidity,
        "ldr_value":     ldr_value,
        "soil_moisture": soil_moisture,
        "distance":      distance,
        "relay1_pump1":  rules["relay1"],
        "relay2_feeder": 0,
        "relay3_pump2":  rules["relay3"],
        "relay4_light":  ml_light
    })

@app.route("/latest")
def latest():
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    if not row:
        return jsonify({"error": "No data yet — call /demo first"}), 404

    if isinstance(row.get("timestamp"), datetime):
        row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return jsonify(row)

@app.route("/sensor-data")
def sensor_data():
    limit = min(int(request.args.get("limit", 100)), 1000)
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM sensor_data ORDER BY id DESC LIMIT %s", (limit,)
        )
        rows = cursor.fetchall()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(ts_fix(rows))

@app.route("/chart-data")
def chart_data():
    points = min(int(request.args.get("points", 50)), 200)
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM (
                SELECT * FROM sensor_data ORDER BY id DESC LIMIT %s
            ) sub ORDER BY id ASC
        """, (points,))
        rows = cursor.fetchall()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(ts_fix(rows))

@app.route("/export/csv")
def export_csv():
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sensor_data ORDER BY id ASC")
        rows = cursor.fetchall()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "temperature", "humidity",
                     "ldr_value", "soil_moisture", "distance",
                     "relay1_pump1", "relay2_feeder",
                     "relay3_pump2", "relay4_light"])
    for r in rows:
        ts = r["timestamp"]
        if isinstance(ts, datetime):
            ts = ts.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([r["id"], ts, r["temperature"], r["humidity"],
                         r["ldr_value"], r["soil_moisture"], r["distance"],
                         r["relay1_pump1"], r["relay2_feeder"],
                         r["relay3_pump2"], r["relay4_light"]])

    filename = f"smartfarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting SmartFarm v2.0 on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
