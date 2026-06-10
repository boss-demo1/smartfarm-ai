"""
SmartFarm IoT — Final Backend v4.0 (Fixed & Optimized)
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
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, date, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error as MySQLError
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── File paths ────────────────────────────────────────────────
BASE_DIR               = os.path.dirname(__file__)
MODEL_PATH             = os.path.join(BASE_DIR, "smartfarm_model.pkl")
IRRIGATION_MODEL_PATH  = os.path.join(BASE_DIR, "irrigation_model.pkl")
IRRIGATION_SCALER_PATH = os.path.join(BASE_DIR, "irrigation_scaler.pkl")
YIELD_MODEL_PATH       = os.path.join(BASE_DIR, "yield_model_last2.pkl")
YIELD_SCALER_PATH      = os.path.join(BASE_DIR, "yield_scaler_last2.pkl")

# ── Constants ─────────────────────────────────────────────────
RETRAIN_EVERY  = 100
TANK_LOW_CM    = 15.0
SOIL_DRY_VAL   = 2500
LDR_DARK_VAL   = 1500
PUMP_MIN_TIME  = 0
PUMP_MAX_TIME  = 30

# ── GDD / Yield constants ─────────────────────────────────────
GDD_BASE      = 10.0
GDD_UPPER     = 30.0
GDD_FLOWERING = 850
GDD_POD_FILL  = 1000
GDD_MATURITY  = 1450
TEMP_OPT_LOW  = 18.0
TEMP_OPT_HIGH = 29.0

# ── Global Model Locks & States ───────────────────────────────
model_state = {
    "model":          None,
    "trained_on":     0,
    "last_retrained": None,
    "accuracy":       None,
    "lock":           threading.Lock()
}

irrigation_model = None
irrigation_scaler = None
irrigation_lock = threading.Lock()

yield_model = None
yield_scaler = None
yield_lock = threading.Lock()

# ── Load all models at startup ────────────────────────────────
try:
    model_state["model"] = joblib.load(MODEL_PATH)
    log.info("Light ML model loaded -> %s", MODEL_PATH)
except Exception:
    log.warning("No light model found — rule fallback active until first retrain")

try:
    irrigation_model  = joblib.load(IRRIGATION_MODEL_PATH)
    irrigation_scaler = joblib.load(IRRIGATION_SCALER_PATH)
    log.info("Irrigation model loaded")
except Exception:
    log.warning("Irrigation model could not be loaded — will train on startup")

try:
    yield_model  = joblib.load(YIELD_MODEL_PATH)
    yield_scaler = joblib.load(YIELD_SCALER_PATH)
    log.info("Yield model loaded")
except Exception:
    log.warning("Yield model could not be loaded — will train on startup")

# ── Database Connection (Fixed syntax errors) ─────────────────
def get_db():
    return mysql.connector.connect(
        host               = os.environ.get("DB_HOST", 'centerbeam.proxy.rlwy.net'),
        user               = os.environ.get("DB_USER", 'root'),
        password           = os.environ.get("DB_PASSWORD", 'rQjNEHvAHmbqgFfnvRZnZRpPcEnBctqd'),
        database           = os.environ.get("DB_NAME", 'railway'),
        port               = int(os.environ.get("DB_PORT", 18813)),
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
        db = get_db(); cursor = db.cursor()
        cursor.execute(sql); db.commit(); db.close()
        log.info("Database table verified/ready")
    except Exception as e:
        log.error("DB init failed: %s", e)

# ── Rule-based relay decisions ────────────────────────────────
def apply_rules(ldr: int, soil: int, distance: float) -> dict:
    return {
        "relay1": 1 if (distance > 0 and distance > TANK_LOW_CM) else 0,
        "relay2": 0,
        "relay3": 1 if soil > SOIL_DRY_VAL else 0,
        "relay4": 1 if ldr  < LDR_DARK_VAL else 0,
        "servo":  "follows relay2 timer"
    }

# ── Light model — auto retrain ────────────────────────────────
def retrain_model():
    try:
        log.info("[RETRAIN] Pulling rows from MySQL...")
        db = get_db(); cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT temperature, humidity, ldr_value,
                   soil_moisture, distance, relay3_light
            FROM sensor_data
        """)
        rows = cursor.fetchall(); db.close()

        if len(rows) < 20:
            log.warning("[RETRAIN] Not enough rows (%d) — need 20+", len(rows))
            return

        df = pd.DataFrame(rows)
        X  = df[["temperature", "humidity", "ldr_value", "soil_moisture", "distance"]]
        y  = df["relay3_light"]

        new_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
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
    try:
        db = get_db(); cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_data")
        total = cursor.fetchone()[0]; db.close()
        new_rows = total - model_state["trained_on"]
        if new_rows >= RETRAIN_EVERY:
            log.info("[RETRAIN] Triggered — %d new rows", new_rows)
            threading.Thread(target=retrain_model, daemon=True).start()
    except Exception as e:
        log.error("[RETRAIN CHECK] %s", e)

# ── Light prediction ──────────────────────────────────────────
def predict_light(temperature, humidity, ldr, soil, distance):
    with model_state["lock"]:
        current_model = model_state["model"]
    if current_model is not None:
        df = pd.DataFrame([{
            "temperature": temperature, "humidity": humidity,
            "ldr_value": ldr, "soil_moisture": soil, "distance": distance
        }])
        return int(current_model.predict(df)[0])
    return 1 if ldr < LDR_DARK_VAL else 0

# ── Irrigation training (Handles 40k+ real rows smoothly) ─────
def train_irrigation_model_from_db():
    global irrigation_model, irrigation_scaler
    try:
        db = get_db(); cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT temperature, humidity, ldr_value, soil_moisture, distance FROM sensor_data")
        rows = cursor.fetchall(); db.close()

        np.random.seed(42)
        
        if len(rows) < 50:
            log.warning("[IRRIGATION TRAIN] Using synthetic data due to empty/small DB")
            n = 500
            soil = np.random.randint(500, 3000, n)
            temp = np.random.uniform(15, 40, n)
            hum  = np.random.uniform(30, 95, n)
            light= np.random.randint(0, 1023, n)
            wlvl = np.random.randint(0, 100, n)
            tday = np.random.randint(0, 24, n)
            pump = ((1 - soil/3000)*18 + ((temp-20)/20)*8 - (hum/100)*6
                    + np.random.normal(0, 2, n)).clip(PUMP_MIN_TIME, PUMP_MAX_TIME)
            df = pd.DataFrame({'soil': soil, 'temp': temp, 'humidity': hum,
                                'light': light, 'water_level': wlvl,
                                'time_day': tday, 'pump_time': pump})
        else:
            log.info("[IRRIGATION TRAIN] Training on %d real database rows!", len(rows))
            df = pd.DataFrame(rows).rename(columns={
                'temperature': 'temp', 'ldr_value': 'light',
                'soil_moisture': 'soil', 'distance': 'water_level'
            })
            df['time_day'] = datetime.now().hour
            df['pump_time'] = (
                (1 - df['soil']/3000)*18 + ((df['temp']-20)/20)*8 - (df['humidity']/100)*6
            ).clip(PUMP_MIN_TIME, PUMP_MAX_TIME)

        FEATURES = ['soil', 'temp', 'humidity', 'light', 'water_level', 'time_day']
        X = df[FEATURES]; y = df['pump_time']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        m = RandomForestRegressor(n_estimators=200, max_depth=15,
                                  min_samples_leaf=3, random_state=42, n_jobs=-1)
        m.fit(X_tr_sc, y_tr)
        
        log.info("[IRRIGATION TRAIN] MAE=%.2f  R²=%.4f",
                 mean_absolute_error(y_te, m.predict(X_te_sc)),
                 r2_score(y_te, m.predict(X_te_sc)))

        joblib.dump(m, IRRIGATION_MODEL_PATH)
        joblib.dump(scaler, IRRIGATION_SCALER_PATH)
        
        with irrigation_lock:
            irrigation_model = m
            irrigation_scaler = scaler
        log.info("[IRRIGATION TRAIN] Saved successfully")
    except Exception as e:
        log.error("[IRRIGATION TRAIN] Failed: %s", e)

# ── Yield training ────────────────────────────────────────────
def train_yield_model():
    global yield_model, yield_scaler
    try:
        log.info("[YIELD TRAIN] Training with tracking constraints...")
        np.random.seed(42); n = 120

        df = pd.DataFrame({
            "day_number":  np.arange(1, n + 1),
            "soil":        np.random.randint(700, 2600, n),
            "temp":        np.random.uniform(18, 35, n),
            "humidity":    np.random.uniform(35, 90, n),
            "light":       np.random.randint(300, 1000, n),
            "water_level": np.random.randint(40, 100, n),
            "pump_time":   np.random.uniform(5, 25, n)
        })

        df["gdd_daily"]                  = df["temp"].apply(lambda t: max(0, min(t, GDD_UPPER) - GDD_BASE))
        df["cumulative_gdd"]             = df["gdd_daily"].cumsum()
        df["cumulative_water"]           = df["pump_time"].cumsum()
        df["temp_stress_daily"]          = df["temp"].apply(
            lambda t: max(0, TEMP_OPT_LOW - t) + max(0, t - TEMP_OPT_HIGH))
        df["cumulative_temp_stress"]     = df["temp_stress_daily"].cumsum()
        df["humidity_stress_day"]        = (df["humidity"] > 80).astype(int)
        df["cumulative_humidity_stress"] = df["humidity_stress_day"].cumsum()
        df["cumulative_light"]           = df["light"].cumsum()

        soil_s  = (1 - abs(df['soil']     - 1500) / 1500).clip(0,1)
        temp_s  = (1 - abs(df['temp']     - 24)   / 15  ).clip(0,1)
        hum_s   = (1 - abs(df['humidity'] - 60)   / 60  ).clip(0,1)
        light_s = (df['light'] / 1000).clip(0,1)
        water_s = (1 - abs(df['pump_time']- 15)   / 15  ).clip(0,1)
        stress  = (df['cumulative_temp_stress'] / (df['cumulative_temp_stress'].max()+1)).clip(0,1)
        growth  = (df['cumulative_gdd'] / GDD_MATURITY).clip(0,1)

        df['yield_pods']  = (2 + soil_s*4 + temp_s*5 + hum_s*2 + light_s*3
                             + water_s*5 + growth*8 - stress*5
                             + np.random.normal(0,1.5,n)).clip(1,30).round(1)
        df['yield_grams'] = (df['yield_pods']*0.6 + np.random.normal(0,0.5,n)).clip(1,20).round(2)

        YFEAT = ['cumulative_gdd','cumulative_water','cumulative_temp_stress',
                 'cumulative_humidity_stress','cumulative_light',
                 'temp','humidity','soil','light','water_level','day_number']

        X = df[YFEAT]; y = df['yield_grams']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=4, min_samples_leaf=3,
                                      subsample=0.8, random_state=42)
        m.fit(X_tr_sc, y_tr)
        log.info("[YIELD TRAIN] MAE=%.2f g  R²=%.4f",
                 mean_absolute_error(y_te, m.predict(X_te_sc)),
                 r2_score(y_te, m.predict(X_te_sc)))

        joblib.dump(m, YIELD_MODEL_PATH)
        joblib.dump(scaler, YIELD_SCALER_PATH)
        
        with yield_lock:
            yield_model = m
            yield_scaler = scaler
        log.info("[YIELD TRAIN] Saved successfully")
    except Exception as e:
        log.error("[YIELD TRAIN] Failed: %s", e)

# ── Prediction helpers ────────────────────────────────────────
def predict_pump_time(soil, temp, humidity, light, water_level, time_day):
    with irrigation_lock:
        local_model = irrigation_model
        local_scaler = irrigation_scaler

    if local_model is None or local_scaler is None:
        return 15.0 if soil > SOIL_DRY_VAL else 0.0
        
    X = pd.DataFrame([{'soil': soil, 'temp': temp, 'humidity': humidity,
                        'light': light, 'water_level': water_level, 'time_day': time_day}])
    X_sc = local_scaler.transform(X)
    pump = local_model.predict(X_sc)[0]
    return round(float(np.clip(pump, PUMP_MIN_TIME, PUMP_MAX_TIME)), 1)

YIELD_FEATURES = ['cumulative_gdd','cumulative_water','cumulative_temp_stress',
                  'cumulative_humidity_stress','cumulative_light',
                  'temp','humidity','soil','light','water_level','day_number']

def growth_stage(gdd):
    if gdd < GDD_FLOWERING:  return "Vegetative"
    elif gdd < GDD_POD_FILL: return "Flowering"
    elif gdd < GDD_MATURITY: return "Pod Fill"
    else:                    return "Mature"

def predict_yield_grams(cumulative_gdd, cumulative_water, cumulative_temp_stress,
                        cumulative_humidity_stress, cumulative_light,
                        day_number, temp, humidity, soil, light, water_level):
    with yield_lock:
        local_model = yield_model
        local_scaler = yield_scaler

    if local_model is None or local_scaler is None:
        return None
        
    X = pd.DataFrame([{
        'cumulative_gdd': cumulative_gdd, 'cumulative_water': cumulative_water,
        'cumulative_temp_stress': cumulative_temp_stress,
        'cumulative_humidity_stress': cumulative_humidity_stress,
        'cumulative_light': cumulative_light, 'temp': temp, 'humidity': humidity,
        'soil': soil, 'light': light, 'water_level': water_level, 'day_number': day_number
    }])
    X_sc  = local_scaler.transform(X[YIELD_FEATURES])
    grams = float(np.clip(local_model.predict(X_sc)[0], 10, 30))
    pods  = round(grams / 0.6, 0)
    stage = growth_stage(cumulative_gdd)
    avg_gdd_day    = cumulative_gdd / max(day_number, 1)
    remaining_gdd  = max(0, GDD_MATURITY - cumulative_gdd)
    days_remaining = int(np.ceil(remaining_gdd / avg_gdd_day)) if avg_gdd_day > 0 else 999
    harvest_date   = (date.today() + timedelta(days=days_remaining)).isoformat()
    return {"yield_grams": round(grams,2), "yield_pods": pods,
            "growth_stage": stage, "days_remaining": days_remaining,
            "harvest_date": harvest_date}

def fix_timestamps(rows):
    for r in rows:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    return rows

@app.errorhandler(Exception)
def handle_error(e):
    log.error("Unhandled API Exception: %s", e)
    return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    with model_state["lock"]:
        has_model  = model_state["model"] is not None
        trained_on = model_state["trained_on"]
        last_rt    = model_state["last_retrained"]
        accuracy   = model_state["accuracy"]
    return jsonify({
        "status":   "SmartFarm API v4.0 online",
        "interval": "15 seconds",
        "sensors":  ["DHT22(D4)","LDR(D34)","Soil(D35)","Ultrasonic(D27/D33)"],
        "actuators":["Relay1-Pump1(D14)","Relay2-Pump2(D12)",
                     "Relay3-Light(D13)","Relay4-Feeder(D15)","Servo(D25)"],
        "models": {
            "light_classifier": "loaded" if has_model        else "rule fallback",
            "irrigation":       "loaded" if irrigation_model is not None else "not loaded",
            "yield":            "loaded" if yield_model is not None else "not loaded"
        },
        "light_model": {
            "trained_on":     f"{trained_on} rows",
            "last_retrained": last_rt or "never",
            "accuracy":       f"{accuracy}%" if accuracy else "pending"
        },
        "endpoints": [
            "GET /upload", "GET /predict",
            "GET /predict/irrigation", "GET /predict/yield",
            "GET /latest", "GET /sensor-data", "GET /chart-data",
            "GET /export/csv", "GET /health", "GET /model-status", "GET /demo"
        ],
        "retrain_every": f"{RETRAIN_EVERY} rows (~{RETRAIN_EVERY*15//60} min)"
    })

@app.route("/health")
def health():
    try:
        db = get_db(); db.close()
        with model_state["lock"]:
            has_model = model_state["model"] is not None
        return jsonify({
            "db":               "ok",
            "light_model":      "loaded" if has_model        else "fallback",
            "irrigation_model": "loaded" if irrigation_model is not None else "not loaded",
            "yield_model":      "loaded" if yield_model is not None else "not loaded"
        }), 200
    except Exception as e:
        return jsonify({"db": "error", "detail": str(e)}), 500

@app.route("/model-status")
def model_status():
    try:
        db = get_db(); cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_data")
        total = cursor.fetchone()[0]; db.close()
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
        "light_model_loaded":      has_model,
        "light_accuracy":          f"{accuracy}%" if accuracy else "pending",
        "irrigation_model_loaded": irrigation_model is not None,
        "yield_model_loaded":      yield_model is not None,
        "total_rows_in_db":        total,
        "trained_on_rows":         trained_on,
        "new_rows_since_train":    new_rows,
        "rows_until_retrain":      rows_to_retrain,
        "mins_until_retrain":      mins_to_retrain,
        "last_retrained":          last_rt or "never",
        "retrain_every":           f"{RETRAIN_EVERY} rows"
    })

@app.route("/upload")
def upload():
    try:
        temperature   = float(request.args.get("temperature",  0))
        humidity      = float(request.args.get("humidity",     0))
        ldr_value     = int(request.args.get("ldr_value",      0))
        soil_moisture = int(request.args.get("soil_moisture",  0))
        distance      = float(request.args.get("distance",     0))
        servo_angle   = int(request.args.get("servo_angle",    0))
        relay1        = int(request.args.get("relay1",         0))
        relay2        = int(request.args.get("relay2",         0))
        relay3        = int(request.args.get("relay3",         0))
        relay4        = int(request.args.get("relay4",         0))
        water_level   = int(request.args.get("water_level",    int(distance)))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    ml_light  = predict_light(temperature, humidity, ldr_value, soil_moisture, distance)
    pump_time = predict_pump_time(
        soil=soil_moisture, temp=temperature, humidity=humidity,
        light=ldr_value, water_level=water_level, time_day=datetime.now().hour
    )

    try:
        db = get_db(); cursor = db.cursor()
        cursor.execute("""
            INSERT INTO sensor_data
            (temperature, humidity, ldr_value, soil_moisture,
             distance, servo_angle,
             relay1_pump1, relay2_pump2, relay3_light, relay4_feeder)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (temperature, humidity, ldr_value, soil_moisture,
              distance, servo_angle, relay1, relay2, relay3, relay4))
        db.commit(); new_id = cursor.lastrowid; db.close()
    except MySQLError as e:
        log.error("DB insert failed: %s", e)
        return jsonify({"error": "Database write failed"}), 500

    maybe_retrain()

    log.info("id=%s | t=%.1f h=%.1f ldr=%d soil=%d dist=%.1f "
             "sv=%d r1=%d r2=%d r3=%d r4=%d ml=%d pump=%.1fs",
             new_id, temperature, humidity, ldr_value, soil_moisture,
             distance, servo_angle, relay1, relay2, relay3, relay4, ml_light, pump_time)

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
        "ml_light_predict":  ml_light,
        "pump_time_seconds": pump_time
    })

@app.route("/predict")
def predict():
    try:
        temperature   = float(request.args.get("temperature",  0))
        humidity      = float(request.args.get("humidity",     0))
        ldr_value     = int(request.args.get("ldr_value",      0))
        soil_moisture = int(request.args.get("soil_moisture",  0))
        distance      = float(request.args.get("distance",     0))
        water_level   = int(request.args.get("water_level",    int(distance)))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    rules     = apply_rules(ldr_value, soil_moisture, distance)
    ml_light  = predict_light(temperature, humidity, ldr_value, soil_moisture, distance)
    pump_time = predict_pump_time(
        soil=soil_moisture, temp=temperature, humidity=humidity,
        light=ldr_value, water_level=water_level, time_day=datetime.now().hour
    )

    with model_state["lock"]:
        has_model  = model_state["model"] is not None
        trained_on = model_state["trained_on"]
        accuracy   = model_state["accuracy"]

    return jsonify({
        "inputs": {
            "temperature": temperature, "humidity": humidity,
            "ldr_value": ldr_value, "soil_moisture": soil_moisture,
            "distance_cm": distance
        },
        "decisions": {
            "relay1_pump1":      rules["relay1"],
            "relay2_pump2":      rules["relay2"],
            "relay3_light":      ml_light,
            "relay4_feeder":     "timer — ON 5s every 1min",
            "servo_angle":       rules["servo"],
            "pump_time_seconds": pump_time
        },
        "reasons": {
            "relay1":    f"tank dist {distance}cm {'> 15 → PUMP ON' if rules['relay1'] else '<= 15 → PUMP OFF'}",
            "relay2":    "timer — ON 5s every 1min → servo sweeps to 90°",
            "relay3":    f"soil {soil_moisture} {'> 2500 → DRY → PUMP2 ON' if rules['relay3'] else '<= 2500 → WET → PUMP2 OFF'}",
            "relay4":    f"ldr {ldr_value} {'< 1500 → DARK → LIGHT ON' if rules['relay4'] else '>= 1500 → BRIGHT → LIGHT OFF'}",
            "servo":     "sweeps 0°→90° when feeder relay2 activates every 1min",
            "pump_time": f"irrigation ML → run pump for {pump_time}s"
        },
        "model_used": "ml" if has_model else "rule_fallback",
        "trained_on": f"{trained_on} rows",
        "accuracy":   f"{accuracy}%" if accuracy else "pending"
    })

@app.route("/predict/irrigation")
def predict_irrigation_route():
    try:
        soil        = int(request.args.get("soil",         1500))
        temp        = float(request.args.get("temp",       25))
        humidity    = float(request.args.get("humidity",   60))
        light       = int(request.args.get("light",        500))
        water_level = int(request.args.get("water_level",  70))
        time_day    = int(request.args.get("time_day",     datetime.now().hour))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    pump_time = predict_pump_time(soil, temp, humidity, light, water_level, time_day)

    return jsonify({
        "inputs": {
            "soil": soil, "temp": temp, "humidity": humidity,
            "light": light, "water_level": water_level, "time_day": time_day
        },
        "pump_time_seconds":   pump_time,
        "pump_recommendation": f"Run pump for {pump_time} seconds" if pump_time > 0 else "No irrigation needed",
        "model":               "irrigation_rf" if irrigation_model is not None else "rule_fallback"
    })

@app.route("/predict/yield")
def predict_yield_route():
    try:
        day_number                 = int(request.args.get("day_number",                   1))
        cumulative_gdd             = float(request.args.get("cumulative_gdd",             0))
        cumulative_water           = float(request.args.get("cumulative_water",           0))
        cumulative_temp_stress     = float(request.args.get("cumulative_temp_stress",     0))
        cumulative_humidity_stress = float(request.args.get("cumulative_humidity_stress", 0))
        cumulative_light           = float(request.args.get("cumulative_light",           0))
        temp                       = float(request.args.get("temp",       25))
        humidity                   = float(request.args.get("humidity",   60))
        soil                       = int(request.args.get("soil",         1500))
        light                      = int(request.args.get("light",        500))
        water_level                = int(request.args.get("water_level",  70))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad parameter: {e}"}), 400

    result = predict_yield_grams(
        cumulative_gdd, cumulative_water, cumulative_temp_stress,
        cumulative_humidity_stress, cumulative_light,
        day_number, temp, humidity, soil, light, water_level
    )

    if result is None:
        return jsonify({"error": "Yield model not loaded — try again in a few seconds"}), 503

    return jsonify({
        "inputs": {
            "day_number": day_number, "cumulative_gdd": cumulative_gdd,
            "cumulative_water": cumulative_water,
            "cumulative_temp_stress": cumulative_temp_stress,
            "cumulative_humidity_stress": cumulative_humidity_stress,
            "cumulative_light": cumulative_light,
            "temp": temp, "humidity": humidity,
            "soil": soil, "light": light, "water_level": water_level
        },
        **result,
        "model": "yield_gb"
    })

@app.route("/demo")
def demo():
    import random
    temperature   = round(random.uniform(20, 35), 2)
    humidity      = round(random.uniform(40, 80), 2)
    ldr_value     = random.randint(0, 4095)
    soil_moisture = random.randint(1000, 4000)
    distance      = round(random.uniform(2, 40), 2)
    servo_angle   = 90 if soil_moisture > SOIL_DRY_VAL else 0
    water_level   = int(distance)

    rules     = apply_rules(ldr_value, soil_moisture, distance)
    ml_light  = predict_light(temperature, humidity, ldr_value, soil_moisture, distance)
    pump_time = predict_pump_time(soil_moisture, temperature, humidity,
                                  ldr_value, water_level, datetime.now().hour)

    try:
        db = get_db(); cursor = db.cursor()
        cursor.execute("""
            INSERT INTO sensor_data
            (temperature, humidity, ldr_value, soil_moisture,
             distance, servo_angle,
             relay1_pump1, relay2_pump2, relay3_light, relay4_feeder)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (temperature, humidity, ldr_value, soil_moisture,
              distance, servo_angle, rules["relay1"], rules["relay2"], ml_light, 0))
        db.commit(); db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    maybe_retrain()

    return jsonify({
        "status":            "demo row inserted",
        "temperature":       temperature,
        "humidity":          humidity,
        "ldr_value":         ldr_value,
        "soil_moisture":     soil_moisture,
        "distance":          distance,
        "servo_angle":       servo_angle,
        "relay1_pump1":      rules["relay1"],
        "relay2_pump2":      rules["relay2"],
        "relay3_light":      ml_light,
        "relay4_feeder":     0,
        "pump_time_seconds": pump_time
    })

@app.route("/latest")
def latest():
    try:
        db = get_db(); cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone(); db.close()
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
        db = get_db(); cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT %s", (limit,))
        rows = cursor.fetchall(); db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(fix_timestamps(rows))

@app.route("/chart-data")
def chart_data():
    points = min(int(request.args.get("points", 50)), 200)
    try:
        db = get_db(); cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM (
                SELECT * FROM sensor_data ORDER BY id DESC LIMIT %s
            ) sub ORDER BY id ASC
        """, (points,))
        rows = cursor.fetchall(); db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(fix_timestamps(rows))

@app.route("/export/csv")
def export_csv():
    try:
        db = get_db(); cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sensor_data ORDER BY id ASC")
        rows = cursor.fetchall(); db.close()
    except MySQLError as e:
        return jsonify({"error": str(e)}), 500

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","timestamp","temperature","humidity","ldr_value",
                     "soil_moisture","distance","servo_angle",
                     "relay1_pump1","relay2_pump2","relay3_light","relay4_feeder"])
    for r in rows:
        ts = r["timestamp"]
        if isinstance(ts, datetime):
            ts = ts.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([r["id"], ts, r["temperature"], r["humidity"],
                         r["ldr_value"], r["soil_moisture"], r["distance"],
                         r["servo_angle"], r["relay1_pump1"], r["relay2_pump2"],
                         r["relay3_light"], r["relay4_feeder"]])

    filename = f"smartfarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})

# ── Startup ───────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    # Model training initializes dynamically in isolated threads to prevent startup blocks
    log.info("Spawning background threads for initial model compilation...")
    threading.Thread(target=train_irrigation_model_from_db, daemon=True).start()
    threading.Thread(target=train_yield_model, daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    log.info("SmartFarm v4.0 starting on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
