import os
import random
from datetime import datetime, timedelta
import pymysql
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# ML ENGINE INITIALIZATION LAYER
# -----------------------------------------------------------------------------
try:
    irrigation_model = joblib.load('irrigation_model.pkl')
    irrigation_scaler = joblib.load('irrigation_scaler.pkl')
    yield_model = joblib.load('yield_model.pkl')
    print("🚀 All ML Models and Scalers loaded successfully into memory!")
except Exception as e:
    print(f"⚠️ Error loading ML model binaries: {e}")
    irrigation_model, irrigation_scaler, yield_model = None, None, None

# Database connection helper (Configured for your Railway Environment Variables)
def get_db_connection():
    return pymysql.connect(
        host=os.getenv('MYSQLHOST', 'centerbeam.proxy.rlwy.net'),
        user=os.getenv('MYSQLUSER', 'root'),
        password=os.getenv('MYSQLPASSWORD', 'rQjNEHvAHmbqgFfnvRZnZRpPcEnBctqd'),
        database=os.getenv('MYSQLDATABASE', 'railway'),
        port=int(os.getenv('MYSQLPORT', 18813)),
        cursorclass=pymysql.cursors.DictCursor
    )

# -----------------------------------------------------------------------------
# DIAGNOSTIC HEALTH ROUTE
# -----------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "database_connected": False,
        "models_loaded": False,
        "error_logs": None
    }
    
    if irrigation_model is not None and yield_model is not None:
        health_status["models_loaded"] = True
        
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        connection.close()
        health_status["database_connected"] = True
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error_logs"] = f"Database connection failed: {str(e)}"
        
    return jsonify(health_status)

# -----------------------------------------------------------------------------
# FEATURE CALCULATOR ENGINE FOR PLANT TIMES & WEATHER CUMULATIVES
# -----------------------------------------------------------------------------
def calculate_live_metrics():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Recreate structural tables if missing dynamically
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS farm_settings (
                    setting_key VARCHAR(50) PRIMARY KEY,
                    setting_value VARCHAR(50)
                )
            """)
            
            cursor.execute("SELECT setting_value FROM farm_settings WHERE setting_key = 'sowing_date'")
            sowing_res = cursor.fetchone()
            
            if not sowing_res:
                # Seed a stable baseline sowing date 20 days ago if it's an empty database instance
                sowing_val = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
                cursor.execute("INSERT INTO farm_settings (setting_key, setting_value) VALUES ('sowing_date', %s)", (sowing_val,))
                connection.commit()
                day_number = 20.0
            else:
                sowing_date = datetime.strptime(sowing_res['setting_value'], '%Y-%m-%d')
                day_number = max(1.0, (datetime.now() - sowing_date).total_seconds() / 86400.0)
            
            cursor.execute("SHOW TABLES LIKE 'sensor_data'")
            if not cursor.fetchone():
                return day_number, day_number * 12.5, 2.0, day_number * 15.0

            cursor.execute("SELECT timestamp, temperature, soil_moisture FROM sensor_data ORDER BY timestamp ASC")
            logs = cursor.fetchall()
            
            if not logs:
                return day_number, day_number * 12.5, 2.0, day_number * 15.0
            
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            if df.empty:
                return day_number, day_number * 12.5, 2.0, day_number * 15.0

            # Calculate Cumulative Growing Degree Days (GDD)
            df['date_only'] = df['timestamp'].dt.date
            daily_groups = df.groupby('date_only')
            cumulative_gdd = 0.0
            for _, group in daily_groups:
                if not group.empty:
                    daily_avg = (group['temperature'].max() + group['temperature'].min()) / 2.0
                    cumulative_gdd += max(0.0, daily_avg - 10.0)
                
            # Compute hours elapsed between timeline events securely
            df['time_delta_hours'] = df['timestamp'].diff().dt.total_seconds().fillna(0.0) / 3600.0
            df.loc[df['time_delta_hours'] < 0, 'time_delta_hours'] = 0.0
            df.loc[df['time_delta_hours'] > 12, 'time_delta_hours'] = 2.0
            
            cumulative_heat_stress = float(df.loc[df['temperature'] > 30.0, 'time_delta_hours'].sum())
            cumulative_water = float(((4095 - df['soil_moisture']) * df['time_delta_hours']).sum() / 1000.0)
            
            return day_number, max(10.0, cumulative_gdd), cumulative_heat_stress, max(5.0, cumulative_water)
            
    except Exception as e:
        print(f"Error calculating live cumulative features: {e}")
        return 20.0, 250.0, 2.0, 100.0
    finally:
        if 'connection' in locals():
            connection.close()

# -----------------------------------------------------------------------------
# CORE ML PROCESSING LOGIC LAYER
# -----------------------------------------------------------------------------
def execute_ml_pipeline(temp, hum, ldr, soil, ultrasonic):
    """Internal core engine to feed custom parameters straight into machine learning files."""
    day_number, cum_gdd, heat_stress, cum_water = calculate_live_metrics()
    
    # 1. Machine Learning Pipeline: Irrigation Prediction
    ml_predicted_duration = 0.0
    if irrigation_model and irrigation_scaler:
        try:
            input_data_irr = [[temp, hum, soil, day_number]]
            scaled_features_irr = irrigation_scaler.transform(input_data_irr)
            ml_predicted_duration = float(irrigation_model.predict(scaled_features_irr)[0])
        except Exception as e:
            print(f"Irrigation Model evaluation error: {e}")
            ml_predicted_duration = 15.0  # Safe execution fallback
    
    # 2. Apply Rule-Based Hardware Backup Overrides
    pump1_status = 1 if ultrasonic > 10.0 else 0
    pump2_status = 1 if soil > 2500 else 0
    relay4_light = 1 if ldr <= 1500 else 0
    
    # 3. Machine Learning Pipeline: Yield Forecasting Calculation
    predicted_yield_value = 0.0
    if yield_model:
        try:
            input_data_yield = [[day_number, cum_gdd, heat_stress, cum_water]]
            raw_yield_preds = yield_model.predict(input_data_yield)
            
            if hasattr(raw_yield_preds, "ndim") and raw_yield_preds.ndim > 1:
                predicted_yield_value = float(raw_yield_preds[0][0])
            elif hasattr(raw_yield_preds, "__len__"):
                predicted_yield_value = float(raw_yield_preds[0])
            else:
                predicted_yield_value = float(raw_yield_preds)
        except Exception as e:
            print(f"Yield Model evaluation error: {e}")
            predicted_yield_value = 1.2
            
    # Biological Growth Stage Categories
    if day_number <= 15:
        growth_stage = "Germination / Emergence"
    elif day_number <= 45:
        growth_stage = "Vegetative Development"
    else:
        growth_stage = "Flowering / Maturation"

    return {
        "sensor_snapshots": {
            "temperature": round(temp, 1),
            "humidity": round(hum, 1),
            "soil_moisture": int(soil),
            "ldr_value": int(ldr),
            "ultrasonic_distance_cm": round(ultrasonic, 1)
        },
        "actuator_states": {
            "pump1_tank_refill": pump1_status,
            "pump2_irrigation_pump": pump2_status,
            "pump2_ml_recommended_duration_seconds": max(0.0, round(ml_predicted_duration, 1)),
            "relay4_growth_light": relay4_light,
            "servo_schedule": "Every 20s rotate 0 to 180 for 3s",
            "ai_lighting_matrix_target": "OPTIMAL MATRIX ZONE" if ldr > 1500 else "BALANCED MATRIX BOOST"
        },
        "agronomic_yield_forecast": {
            "current_growth_stage": growth_stage,
            "accumulated_gdd": round(cum_gdd, 1),
            "expected_yield_grams": round(max(0.0, predicted_yield_value), 1),
            "projected_days_until_harvest": max(0, int(110 - day_number))
        }
    }

# -----------------------------------------------------------------------------
# PRODUCTION PREDICT ENDPOINT PIPELINE (GET / POST)
# -----------------------------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            req_data = request.get_json() or {}
            temp = float(req_data.get('temperature', 25.0))
            hum = float(req_data.get('humidity', 50.0))
            ldr = int(req_data.get('ldr_value', 2000))
            soil = int(req_data.get('soil_moisture', 2000))
            ultrasonic = float(req_data.get('distance', req_data.get('ultrasonic_distance', 5.0)))
        else:
            temp = float(request.args.get('temperature', 25.0))
            hum = float(request.args.get('humidity', 50.0))
            ldr = int(request.args.get('ldr_value', 2000))
            soil = int(request.args.get('soil_moisture', 2000))
            ultrasonic = float(request.args.get('distance', request.args.get('ultrasonic_distance', 5.0)))
        
        # Process data through ML Core Calculations Engine
        response_payload = execute_ml_pipeline(temp, hum, ldr, soil, ultrasonic)
        
        # Append latest reading safely to the analytics table records
        try:
            db_conn = get_db_connection()
            with db_conn.cursor() as db_cursor:
                db_cursor.execute("""
                    INSERT INTO sensor_data 
                    (timestamp, temperature, humidity, ldr_value, soil_moisture, ultrasonic_distance) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp, hum, ldr, soil, ultrasonic))
                db_conn.commit()
            db_conn.close()
        except Exception as db_err:
            print(f"⚠️ Telemetry log storage exception: {db_err}")

        return jsonify(response_payload)
    except Exception as err:
        return jsonify({"error": str(err), "status": "ML Engine processing broken"}), 500

# -----------------------------------------------------------------------------
# FIXED LATEST ROUTE: Serves Data + Computes ML Logic Automatically!
# -----------------------------------------------------------------------------
@app.route('/latest', methods=['GET'])
def get_latest():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'sensor_data'")
            if not cursor.fetchone():
                # If table doesn't exist, return ideal mock metrics directly inside the payload shape
                return jsonify(execute_ml_pipeline(25.0, 52.0, 1800, 1600, 6.2))

            sql = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1"
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if result and isinstance(result, dict):
                # Pass data row parameters directly into the ML evaluation engine
                temp = float(result.get('temperature', 25.0))
                hum = float(result.get('humidity', 50.0))
                ldr = int(result.get('ldr_value', 2000))
                soil = int(result.get('soil_moisture', 2000))
                ultrasonic = float(result.get('ultrasonic_distance', 5.0))
                
                dashboard_payload = execute_ml_pipeline(temp, hum, ldr, soil, ultrasonic)
                return jsonify(dashboard_payload)
            
            # If the database is completely empty, serve a mock payload so the dashboard never renders blank lines
            return jsonify(execute_ml_pipeline(24.5, 55.0, 1200, 1950, 7.0))
    except Exception as e:
        return jsonify({"error": "Dashboard sync compilation down", "details": str(e)}), 500
    finally:
        if 'connection' in locals():
            connection.close()

# -----------------------------------------------------------------------------
# RESTORED ROUTE 2: Fetch historical records for telemetry charts
# -----------------------------------------------------------------------------
@app.route('/sensor-data', methods=['GET'])
def get_sensor_data():
    limit = request.args.get('limit', default=50, type=int)
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'sensor_data'")
            if not cursor.fetchone():
                return jsonify([])

            sql = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT %s"
            cursor.execute(sql, (limit,))
            results = cursor.fetchall()
            
            if results:
                for row in results:
                    if isinstance(row, dict) and 'timestamp' in row and hasattr(row['timestamp'], 'strftime'):
                        row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                return jsonify(results)
            return jsonify([])
    except Exception as e:
        return jsonify({"error": "DB Processing Error", "details": str(e)}), 500
    finally:
        if 'connection' in locals():
            connection.close()

# -----------------------------------------------------------------------------
# TWO-WEEK SEQUENTIAL CHRONOLOGICAL HISTORY GENERATOR
# -----------------------------------------------------------------------------
@app.route('/demo', methods=['GET'])
def generate_demo_data():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Insulate structural requirements
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS farm_settings (
                    setting_key VARCHAR(50) PRIMARY KEY,
                    setting_value VARCHAR(50)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME,
                    temperature FLOAT,
                    humidity FLOAT,
                    ldr_value INT,
                    soil_moisture INT,
                    ultrasonic_distance FLOAT
                )
            """)
            
            # Setup sowing date row entry
            cursor.execute("""
                INSERT INTO farm_settings (setting_key, setting_value) 
                VALUES ('sowing_date', %s) 
                ON DUPLICATE KEY UPDATE setting_value=%s
            """, ((datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d'), (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')))
            
            cursor.execute("TRUNCATE TABLE sensor_data")
            
            temperature = 22.0
            humidity = 65.0
            ldr_value = 1500
            soil_moisture = 1800 
            ultrasonic_distance = 5.0
            
            current_time = datetime.now()
            inserted_count = 0
            
            for hours_back in range(336, -1, -2):
                record_timestamp = current_time - timedelta(hours=hours_back)
                hour = record_timestamp.hour
                
                if 6 <= hour <= 15: 
                    temperature += random.uniform(-0.2, 0.5)
                    humidity += random.uniform(-0.8, 0.2)
                    ldr_value -= random.randint(80, 200)
                else: 
                    temperature += random.uniform(-0.5, 0.2)
                    humidity += random.uniform(-0.2, 0.8)
                    ldr_value += random.randint(80, 200)
                
                temperature = max(17.0, min(36.0, temperature))
                humidity = max(30.0, min(90.0, humidity))
                ldr_value = max(150, min(4095, ldr_value))
                
                if soil_moisture >= 2500:
                    soil_moisture = random.randint(1400, 1700)
                else:
                    evaporation_factor = 35 if temperature > 30.0 else 18
                    soil_moisture += random.randint(5, evaporation_factor)
                
                soil_moisture = max(500, min(3800, soil_moisture))
                
                if ultrasonic_distance > 15.0:
                    ultrasonic_distance = 5.0
                else:
                    ultrasonic_distance += random.uniform(0.02, 0.1)
                
                ultrasonic_distance = round(max(2.0, ultrasonic_distance), 1)
                formatted_ts = record_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                sql = """
                    INSERT INTO sensor_data 
                    (timestamp, temperature, humidity, ldr_value, soil_moisture, ultrasonic_distance) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (formatted_ts, round(temperature, 1), round(humidity, 1), 
                                     int(ldr_value), int(soil_moisture), ultrasonic_distance))
                inserted_count += 1
                
            connection.commit()
            
            return jsonify({
                "status": "success",
                "message": "Successfully generated a robust 14-day historical matrix for your crop!",
                "total_records_inserted": inserted_count,
                "timeline_range": {
                    "start": (current_time - timedelta(days=14)).strftime('%Y-%m-%d %H:%M:%S'),
                    "end": current_time.strftime('%Y-%m-%d %H:%M:%S')
                }
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
