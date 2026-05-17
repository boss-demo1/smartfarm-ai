"""
SmartFarm IoT — Cloud Backend
Flask + MySQL (Railway) + ML Model (scikit-learn)
Deploy target: Render.com

Environment variables required (set in Render dashboard):
  DB_HOST      — Railway MySQL host
  DB_USER      — Railway MySQL user
  DB_PASSWORD  — Railway MySQL password
  DB_NAME      — Railway MySQL database name
  PORT         — auto-set by Render (don't touch)
"""

import os
import io
import csv
import logging
from datetime import datetime

import pandas as pd
import joblib
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error as MySQLError

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── ML Model ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "smartfarm_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    log.info("ML model loaded from %s", MODEL_PATH)
except FileNotFoundError:
    model = None
    log.warning("smartfarm_model.pkl not found — /predict and /upload will return error until model is added")

# ── Database ──────────────────────────────────────────────────
def get_db():
    """
    Open a MySQL connection using environment variables.
    Raises a clear error if any variable is missing.
    """
    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    return mysql.connector.connect(
        host     = os.environ["DB_HOST"],
        user     = os.environ["DB_USER"],
        password = os.environ["DB_PASSWORD"],
        database = os.environ["DB_NAME"],
        port     = int(os.environ.get("DB_PORT", 3306)),
        connection_timeout = 10
    )

def init_db():
    """
    Create the sensor_data table if it does not exist.
    Called once at startup.
    """
    sql = """
        CREATE TABLE IF NOT EXISTS sensor_data (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
            temperature  FLOAT   NOT NULL,
            humidity     FLOAT   NOT NULL,
            ldr_value    INT     NOT NULL,
            light_status TINYINT NOT NULL DEFAULT 0
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

# ── ML helper ─────────────────────────────────────────────────
def predict_light(temperature: float, humidity: float, ldr_value: int) -> int:
    """
    Run ML model prediction.
    Falls back to rule-based logic if model is not loaded.
    """
    if model is not None:
        df = pd.DataFrame([{
            "temperature": temperature,
            "humidity":    humidity,
            "ldr_value":   ldr_value
        }])
        return int(model.predict(df)[0])
    # Fallback rule (matches training logic)
    return 1 if ldr_value >= 3800 else 0

# ── Error handler ─────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_error(e):
    log.error("Unhandled exception: %s", e)
    return jsonify({"error": str(e)}), 500

# ═════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════

# ── Health check ──────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status":  "SmartFarm API online",
        "version": "3.0",
        "model":   "loaded" if model else "not loaded (fallback rule active)"
    })

@app.route("/health")
def health():
    """Render health-check endpoint."""
    try:
        db = get_db()
        db.close()
        return jsonify({"db": "ok", "model": "ok" if model else "fallback"}), 200
    except Exception as e:
        return jsonify({"db": "error", "detail": str(e)}), 500

# ── ESP32 data upload ─────────────────────────────────────────
@app.route("/upload")
def upload():
    """
    GET /upload?temperature=28.5&humidity=64.2&ldr_value=3900

    Called by ESP32 every 60 seconds.
    Runs ML prediction, stores result, returns JSON.
    Using GET so the ESP32 HTTPClient.begin(url) works without
    setting a Content-Type header.
    """
    try:
        temperature = float(request.args.get("temperature", 0))
        humidity    = float(request.args.get("humidity",    0))
        ldr_value   = int(request.args.get("ldr_value",     0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    # Validate ranges
    if not (-10 <= temperature <= 60):
        return jsonify({"error": "temperature out of range"}), 422
    if not (0 <= humidity <= 100):
        return jsonify({"error": "humidity out of range"}), 422
    if not (0 <= ldr_value <= 4095):
        return jsonify({"error": "ldr_value out of range"}), 422

    light_status = predict_light(temperature, humidity, ldr_value)

    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO sensor_data (temperature, humidity, ldr_value, light_status) VALUES (%s, %s, %s, %s)",
            (temperature, humidity, ldr_value, light_status)
        )
        db.commit()
        new_id = cursor.lastrowid
        db.close()
    except MySQLError as e:
        log.error("DB insert failed: %s", e)
        return jsonify({"error": "Database write failed"}), 500

    log.info("Saved id=%s temp=%.1f hum=%.1f ldr=%d light=%d",
             new_id, temperature, humidity, ldr_value, light_status)

    return jsonify({
        "status":                "saved",
        "id":                    new_id,
        "temperature":           temperature,
        "humidity":              humidity,
        "ldr_value":             ldr_value,
        "predicted_light_status": light_status
    })

# ── ML predict only (no DB write) ────────────────────────────
@app.route("/predict")
def predict():
    """
    GET /predict?temperature=28.5&humidity=64.2&ldr_value=3900
    Returns ML prediction without storing anything.
    Useful for testing the model.
    """
    try:
        temperature = float(request.args.get("temperature", 0))
        humidity    = float(request.args.get("humidity",    0))
        ldr_value   = int(request.args.get("ldr_value",     0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    light_status = predict_light(temperature, humidity, ldr_value)

    return jsonify({
        "temperature":           temperature,
        "humidity":              humidity,
        "ldr_value":             ldr_value,
        "predicted_light_status": light_status,
        "model_used":            "ml" if model else "rule_fallback"
    })

# ── Demo data insert ──────────────────────────────────────────
@app.route("/demo")
def demo():
    """
    GET /demo
    Inserts one random row — useful for testing the DB
    and dashboard without an ESP32.
    """
    import random
    temperature  = round(random.uniform(20, 35), 2)
    humidity     = round(random.uniform(40, 80), 2)
    ldr_value    = random.randint(0, 4095)
    light_status = predict_light(temperature, humidity, ldr_value)

    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO sensor_data (temperature, humidity, ldr_value, light_status) VALUES (%s, %s, %s, %s)",
            (temperature, humidity, ldr_value, light_status)
        )
        db.commit()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status":      "demo row inserted",
        "temperature": temperature,
        "humidity":    humidity,
        "ldr_value":   ldr_value,
        "light_status": light_status
    })

# ── Latest reading ────────────────────────────────────────────
@app.route("/latest")
def latest():
    """GET /latest — most recent sensor row."""
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    if not row:
        return jsonify({"error": "No data yet"}), 404

    # Make timestamp JSON-serialisable
    if isinstance(row.get("timestamp"), datetime):
        row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return jsonify(row)

# ── All history ───────────────────────────────────────────────
@app.route("/sensor-data")
def sensor_data():
    """GET /sensor-data?limit=100 — newest first."""
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

    for r in rows:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return jsonify(rows)

# ── Chart data ────────────────────────────────────────────────
@app.route("/chart-data")
def chart_data():
    """GET /chart-data?points=50 — oldest→newest for chart X axis."""
    points = min(int(request.args.get("points", 50)), 200)
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        # Sub-query: take last N rows, then reverse for chronological order
        cursor.execute("""
            SELECT * FROM (
                SELECT * FROM sensor_data ORDER BY id DESC LIMIT %s
            ) sub ORDER BY id ASC
        """, (points,))
        rows = cursor.fetchall()
        db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    for r in rows:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return jsonify(rows)

# ── CSV export ────────────────────────────────────────────────
@app.route("/export/csv")
def export_csv():
    """GET /export/csv — download all data as CSV file."""
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
    writer.writerow(["id", "timestamp", "temperature", "humidity", "ldr_value", "light_status"])

    for r in rows:
        ts = r["timestamp"]
        if isinstance(ts, datetime):
            ts = ts.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([r["id"], ts, r["temperature"],
                         r["humidity"], r["ldr_value"], r["light_status"]])

    filename = f"smartfarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
