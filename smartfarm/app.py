"""
SmartFarm IoT — Final Backend v3.0
Flask + MySQL (Railway) + Auto-retraining ML
Sensors  : DHT22, LDR, Soil Moisture, Ultrasonic
Actuators: Relay x4, Servo Motor
Interval : 15 seconds
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
RETRAIN_EVERY = 100   # retrain every 100 new rows
                      # 100 rows × 15s = every ~25 minutes

# ── Thresholds — must match ESP32 sketch ─────────────────────
TANK_LOW_CM      = 15.0
SOIL_DRY_VAL     = 2500
LDR_DARK_VAL     = 1500

# ── Model state ───────────────────────────────────────────────
model_state = {
    "model":          None,
    "trained_on":     0,
    "last_retrained": None,
    "accuracy":       None,
    "lock":           threading.Lock()
}

try:
    model_state["model"] = joblib.load(MODEL_PATH)
    log.info("ML model loaded → %s", MODEL_PATH)
except FileNotFoundError:
    log.warning("No model found — rule fallback active until first retrain")

# ── Database ──────────────────────────────────────────────────
def get_db():
    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")
    return mysql.connector.connect(
        host               = os.environ["DB_HOST"],
        user               = os.environ["DB_USER"],
        password           = os.environ["DB_PASSWORD"],
        database           = os.environ["DB_NAME"],
        port               = int(os.environ.get("DB_PORT", 3306)),
        connection_timeout = 10
    )

def init_db():
    sql = """
        CREATE TABLE IF NOT EXISTS sensor_data (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            timestamp     DATETIME   NOT NULL DEFAULT CURRENT_TIMESTAMP,
            temperature   FLOAT      NOT NULL,
            humidity      FLOAT      NOT NULL,
            ldr_value     INT        NOT NULL,
            soil_moisture INT        NOT NULL DEFAULT 0,
            distance      FLOAT      NOT NULL DEFAULT 0,
            servo_angle   INT        NOT NULL DEFAULT 0,
            relay1_pump1  TINYINT(1) NOT NULL DEFAULT 0,
            relay2_pump2  TINYINT(1) NOT NULL DEFAULT 0,
            relay3_light  TINYINT(1) NOT NULL DEFAULT 0,
            relay4_feeder TINYINT(1) NOT NULL DEFAULT 0
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
        log.error("DB init failed: %s", e)

# ── Rule-based decisions ──────────────────────────────────────
def apply_rules(ldr: int, soil: int, distance: float,
                feeder_active: bool = False) -> dict:
    return {
        "relay1": 1 if (distance > 0 and distance > TANK_LOW_CM) else 0,
        "relay2": 1 if soil > SOIL_DRY_VAL else 0,
        "relay3": 1 if ldr  < LDR_DARK_VAL else 0,
        "relay4": 1 if feeder_active else 0,
        "servo":  90 if soil > SOIL_DRY_VAL else 0
    }

# ── ML retrain ────────────────────────────────────────────────
def retrain_model():
    """
    Pull all rows from MySQL and retrain the Random Forest.
    Runs in a background thread — never blocks the API.
    After 100 new rows (100 × 15s = ~25 min) this triggers automatically.
    """
    try:
        log.info("[RETRAIN] Starting — pulling all rows from MySQL...")

        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT temperature, humidity, ldr_value,
                   soil_moisture, distance, relay3_light
            FROM sensor_data
        """)
        rows = cursor.fetchall()
        db.close()

        if len(rows) < 20:
            log.warning("[RETRAIN] Not enough rows (%d) — need 20+", len(rows))
            return

        df = pd.DataFrame(rows)
        X  = df[["temperature", "humidity", "ldr_value",
                  "soil_moisture", "distance"]]
        y  = df["relay3_light"]

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
            model_state["accuracy"]       = accuracy

        log.info("[RETRAIN] Done — %d rows — accuracy %.2f%%", len(rows), accuracy)

    except Exception as e:
        log.error("[RETRAIN] Failed: %s", e)

def maybe_retrain():
    """Check if enough new rows to trigger retrain."""
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_data")
        total = cursor.fetchone()[0]
        db.close()

        new_rows = total - model_state["trained_on"]
        if new_rows >= RETRAIN_EVERY:
            log.info("[RETRAIN] Triggered — %d new rows", new_rows)
            t = threading.Thread(target=retrain_model, daemon=True)
            t.start()
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

    # Rule fallback before first model is trained
    return 1 if ldr < LDR_DARK_VAL else 0

def fix_timestamps(rows):
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
        accuracy   = model_state["accuracy"]
    return jsonify({
        "status":         "SmartFarm API v3.0 online",
        "interval":       "15 seconds",
        "sensors":        ["DHT22(D4)", "LDR(D34)",
                           "Soil(D35)", "Ultrasonic(D27/D33)"],
        "actuators":      ["Relay1-Pump1(D14)", "Relay2-Pump2(D12)",
                           "Relay3-Light(D13)", "Relay4-Feeder(D15)",
                           "Servo(D25)"],
        "model":          "loaded" if has_model else "rule fallback",
        "trained_on":     f"{trained_on} rows",
        "last_retrained": last_rt   or "never",
        "accuracy":       f"{accuracy}%" if accuracy else "pending",
        "retrain_every":  f"{RETRAIN_EVERY} rows (~{RETRAIN_EVERY * 15 // 60} min)"
    })

@app.route("/health")
def health():
    try:
        db = get_db()
        db.close()
        with model_state["lock"]:
            has_model = model_state["model"] is not None
        return jsonify({
            "db":    "ok",
            "model": "loaded" if has_model else "fallback"
        }), 200
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
        accuracy   = model_state["accuracy"]

    new_rows        = total - trained_on
    rows_to_retrain = RETRAIN_EVERY - (new_rows % RETRAIN_EVERY)
    mins_to_retrain = (rows_to_retrain * 15) // 60

    return jsonify({
        "model_loaded":         has_model,
        "accuracy":             f"{accuracy}%" if accuracy else "pending",
        "total_rows_in_db":     total,
        "trained_on_rows":      trained_on,
        "new_rows_since_train": new_rows,
        "rows_until_retrain":   rows_to_retrain,
        "mins_until_retrain":   mins_to_retrain,
        "last_retrained":       last_rt or "never",
        "retrain_every":        f"{RETRAIN_EVERY} rows"
    })

@app.route("/upload")
def upload():
    """
    GET /upload?temperature=&humidity=&ldr_value=&soil_moisture=
               &distance=&servo_angle=&relay1=&relay2=&relay3=&relay4=

    Called by ESP32 every 15 seconds.
    Saves all sensor values + relay states.
    ML predicts light status independently.
    Triggers retrain every 100 new rows (~25 min).
    """
    try:
        temperature   = float(request.args.get("temperature",   0))
        humidity      = float(request.args.get("humidity",      0))
        ldr_value     = int(request.args.get("ldr_value",       0))
        soil_moisture = int(request.args.get("soil_moisture",   0))
        distance      = float(request.args.get("distance",      0))
        servo_angle   = int(request.args.get("servo_angle",     0))
        relay1        = int(request.args.get("relay1",          0))
        relay2        = int(request.args.get("relay2",          0))
        relay3        = int(request.args.get("relay3",          0))
        relay4        = int(request.args.get("relay4",          0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    # ML independently predicts light status
    ml_light = predict_light(temperature, humidity,
                             ldr_value, soil_moisture, distance)

    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO sensor_data
            (temperature, humidity, ldr_value, soil_moisture,
             distance, servo_angle,
             relay1_pump1, relay2_pump2, relay3_light, relay4_feeder)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (temperature, humidity, ldr_value, soil_moisture,
              distance, servo_angle,
              relay1, relay2, relay3, relay4))
        db.commit()
        new_id = cursor.lastrowid
        db.close()
    except MySQLError as e:
        log.error("DB insert failed: %s", e)
        return jsonify({"error": "Database write failed"}), 500

    maybe_retrain()

    log.info("id=%s | t=%.1f h=%.1f ldr=%d soil=%d dist=%.1f "
             "sv=%d° r1=%d r2=%d r3=%d r4=%d ml=%d",
             new_id, temperature, humidity, ldr_value,
             soil_moisture, distance, servo_angle,
             relay1, relay2, relay3, relay4, ml_light)

    return jsonify({
        "status":            "saved",
        "id":                new_id,
        "temperature":       temperature,
        "humidity":          humidity,
        "ldr_value":         ldr_value,
        "soil_moisture":     soil_moisture,
        "distance":          distance,
        "servo_angle":       servo_angle,
        "relay1_pump1":      relay1,
        "relay2_pump2":      relay2,
        "relay3_light":      relay3,
        "relay4_feeder":     relay4,
        "ml_light_predict":  ml_light
    })

@app.route("/predict")
def predict():
    """
    GET /predict?temperature=&humidity=&ldr_value=
                 &soil_moisture=&distance=
    Returns ML prediction + all relay decisions with reasons.
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
    ml_light = predict_light(temperature, humidity,
                             ldr_value, soil_moisture, distance)

    with model_state["lock"]:
        has_model  = model_state["model"] is not None
        trained_on = model_state["trained_on"]
        accuracy   = model_state["accuracy"]

    return jsonify({
        "inputs": {
            "temperature":   temperature,
            "humidity":      humidity,
            "ldr_value":     ldr_value,
            "soil_moisture": soil_moisture,
            "distance_cm":   distance
        },
        "decisions": {
            "relay1_pump1":  rules["relay1"],
            "relay2_pump2":  rules["relay2"],
            "relay3_light":  ml_light,
            "relay4_feeder": "timer — ON 5s every 1min",
            "servo_angle":   rules["servo"]
        },
        "reasons": {
            "relay1": f"tank dist {distance}cm "
                      f"{'< 10 → PUMP ON' if rules['relay1'] else '<= 15 → PUMP OFF'}",
            "relay2": f"soil {soil_moisture} "
                      f"{'> 2500 → DRY → PUMP ON' if rules['relay2'] else '<= 2500 → WET → PUMP OFF'}",
            "relay3": f"ldr {ldr_value} "
                      f"{'< 1500 → DARK → LIGHT ON' if ml_light else '>= 1500 → BRIGHT → LIGHT OFF'}",
            "relay4": "ESP32 timer — independent",
            "servo":  f"soil {soil_moisture} "
                      f"{'> 2500 → 90° open' if rules['servo'] == 90 else '<= 2500 → 0° closed'}"
        },
        "model_used":   "ml" if has_model else "rule_fallback",
        "trained_on":   f"{trained_on} rows",
        "accuracy":     f"{accuracy}%" if accuracy else "pending"
    })

@app.route("/demo")
def demo():
    """Insert one random row — for testing without ESP32."""
    import random
    temperature   = round(random.uniform(20, 35), 2)
    humidity      = round(random.uniform(40, 80), 2)
    ldr_value     = random.randint(0, 4095)
    soil_moisture = random.randint(1000, 4000)
    distance      = round(random.uniform(2, 40), 2)
    servo_angle   = 90 if soil_moisture > SOIL_DRY_VAL else 0
    rules         = apply_rules(ldr_value, soil_moisture, distance)
    ml_light      = predict_light(temperature, humidity,
                                  ldr_value, soil_moisture, distance)

    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO sensor_data
            (temperature, humidity, ldr_value, soil_moisture,
             distance, servo_angle,
             relay1_pump1, relay2_pump2, relay3_light, relay4_feeder)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (temperature, humidity, ldr_value, soil_moisture,
              distance, servo_angle,
              rules["relay1"], rules["relay2"], ml_light, 0))
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
        "servo_angle":   servo_angle,
        "relay1_pump1":  rules["relay1"],
        "relay2_pump2":  rules["relay2"],
        "relay3_light":  ml_light,
        "relay4_feeder": 0
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

    return jsonify(fix_timestamps(rows))

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

    return jsonify(fix_timestamps(rows))

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
    writer.writerow([
        "id", "timestamp", "temperature", "humidity",
        "ldr_value", "soil_moisture", "distance", "servo_angle",
        "relay1_pump1", "relay2_pump2", "relay3_light", "relay4_feeder"
    ])
    for r in rows:
        ts = r["timestamp"]
        if isinstance(ts, datetime):
            ts = ts.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            r["id"], ts, r["temperature"], r["humidity"],
            r["ldr_value"], r["soil_moisture"], r["distance"], r["servo_angle"],
            r["relay1_pump1"], r["relay2_pump2"],
            r["relay3_light"], r["relay4_feeder"]
        ])

    filename = f"smartfarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    log.info("SmartFarm v3.0 starting on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
