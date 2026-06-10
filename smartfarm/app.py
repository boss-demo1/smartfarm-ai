import os
import pickle
import numpy as np
import pandas as pd
import pymysql
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta  # 🚀 FIXED: Added explicit timedelta import here!

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# DATABASE CONNECTION FACTORY
# -----------------------------------------------------------------------------
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
# ML ENGINE INITIALIZATION LAYER
# -----------------------------------------------------------------------------
try:
    with open('irrigation_model.pkl', 'rb') as f:
        irrigation_model = pickle.load(f)
    with open('irrigation_scaler.pkl', 'rb') as f:
        irrigation_scaler = pickle.load(f)
    with open('yield_model.pkl', 'rb') as f:
        yield_model = pickle.load(f)
    print("✅ [ML ENGINE] All predictive models and feature scalers loaded successfully!")
except Exception as e:
    print(f"❌ [ML ENGINE] Critical Initialization Failure: {e}")
    irrigation_model, irrigation_scaler, yield_model = None, None, None

# -----------------------------------------------------------------------------
# FEATURE EXTRACTION ENGINE (MATHEMATICAL CALCULATIONS)
# -----------------------------------------------------------------------------
def calculate_live_metrics():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Verify farm configuration metadata settings exist
            cursor.execute("SHOW TABLES LIKE 'farm_settings'")
            if not cursor.fetchone():
                return 20.0, 250.0, 5.0, 120.0  # Safe mock data fallbacks if tables don't exist
                
            cursor.execute("SELECT setting_value FROM farm_settings WHERE setting_key = 'sowing_date'")
            sowing_res = cursor.fetchone()
            
            if sowing_res:
                sowing_date = datetime.strptime(sowing_res['setting_value'], '%Y-%m-%d')
            else:
                sowing_date = datetime.now() - timedelta(days=20) # Default setup to Day 20 if empty
                
            current_time = datetime.now()
            day_number = max(1.0, (current_time - sowing_date).total_seconds() / 86400.0)
            
            cursor.execute("SHOW TABLES LIKE 'sensor_data'")
            if not cursor.fetchone():
                return day_number, day_number * 12.5, 2.0, day_number * 15.0

            cursor.execute("SELECT timestamp, temperature, soil_moisture FROM sensor_data ORDER BY timestamp ASC")
            logs = cursor.fetchall()
            
            if not logs:
                # Approximate cumulative variables relative to active crop timeline stage
                return day_number, day_number * 12.5, 2.0, day_number * 15.0
            
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            if df.empty:
                return day_number, day_number * 12.5, 2.0, day_number * 15.0

            # Calculate Growing Degree Days (GDD)
            df['date_only'] = df['timestamp'].dt.date
            daily_groups = df.groupby('date_only')
            cumulative_gdd = 0.0
            for _, group in daily_groups:
                daily_avg = (group['temperature'].max() + group['temperature'].min()) / 2.0
                cumulative_gdd += max(0.0, daily_avg - 10.0)
                
            # Safely calculate historical step sizes to prevent timestamp anomalies
            df['time_delta_hours'] = df['timestamp'].diff().dt.total_seconds().fillna(0.0) / 3600.0
            df.loc[df['time_delta_hours'] < 0, 'time_delta_hours'] = 0.0
            df.loc[df['time_delta_hours'] > 12, 'time_delta_hours'] = 2.0
            
            cumulative_heat_stress = float(df.loc[df['temperature'] > 30.0, 'time_delta_hours'].sum())
            cumulative_water = float(((4095 - df['soil_moisture']) * df['time_delta_hours']).sum() / 1000.0)
            
            return day_number, max(10.0, cumulative_gdd), cumulative_heat_stress, max(5.0, cumulative_water)
            
    except Exception as e:
        print(f"⚠️ Calculation Notice: {e}")
        return 20.0, 250.0, 5.0, 120.0
    finally:
        if 'connection' in locals():
            connection.close()

# -----------------------------------------------------------------------------
# UNIFIED ML CORE PREDICT / UPLOAD ROUTE
# -----------------------------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def predict():
    try:
        # Extract fields flexibly checking both standard GET parameters and JSON bodies
        if request.method == 'POST':
            req_data = request.get_json() or {}
            temp = float(req_data.get('temperature', 26.5))
            hum = float(req_data.get('humidity', 55.0))
            ldr = int(req_data.get('ldr_value', 1800))
            soil = int(req_data.get('soil_moisture', 2200))
            ultrasonic = float(req_data.get('distance', req_data.get('ultrasonic_distance', 8.5)))
        else:
            temp = float(request.args.get('temperature', 26.5))
            hum = float(request.args.get('humidity', 55.0))
            ldr = int(request.args.get('ldr_value', 1800))
            soil = int(request.args.get('soil_moisture', 2200))
            ultrasonic = float(request.args.get('distance', request.args.get('ultrasonic_distance', 8.5)))
        
        day_number, cum_gdd, heat_stress, cum_water = calculate_live_metrics()
        
        # ML Inference 1: Irrigation Duration (RandomForestRegressor)
        ml_predicted_duration = 0.0
        if irrigation_model and irrigation_scaler:
            input_data_irr = [[temp, hum, soil, day_number]]
            scaled_features_irr = irrigation_scaler.transform(input_data_irr)
            ml_predicted_duration = float(irrigation_model.predict(scaled_features_irr)[0])
        
        # Rule-based backup hardware indicators
        pump1_status = 1 if ultrasonic > 15.0 else 0
        pump2_status = 1 if soil > 2500 else 0
        relay4_light = 1 if ldr <= 1500 else 0
        
        # ML Inference 2: Crop Yield Forecast
        predicted_yield_value = 0.0
        if yield_model:
            input_data_yield = [[day_number, cum_gdd, heat_stress, cum_water]]
            raw_yield_preds = yield_model.predict(input_data_yield)
            
            # Form factor normalization checks for variations in scikit-learn models outputs
            if hasattr(raw_yield_preds, "ndim") and raw_yield_preds.ndim > 1:
                predicted_yield_value = float(raw_yield_preds[0][0])
            elif hasattr(raw_yield_preds, "__len__"):
                predicted_yield_value = float(raw_yield_preds[0])
            else:
                predicted_yield_value = float(raw_yield_preds)

        # Growth stage determination
        if day_number <= 15:
            growth_stage = "Germination / Emergence"
        elif day_number <= 45:
            growth_stage = "Vegetative Development"
        else:
            growth_stage = "Flowering / Maturation"

        # Log active request data to standard DB records for pipeline persistence
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
            print(f"⚠️ Database tracking logging bypassed: {db_err}")

        # Final standardized JSON response matching your dashboard template
        return jsonify({
            "sensor_snapshots": {
                "temperature": temp,
                "humidity": hum,
                "soil_moisture": soil,
                "ldr_value": ldr,
                "ultrasonic_distance_cm": ultrasonic
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
        })
    except Exception as err:
        return jsonify({"error": str(err), "status": "ML Engine processing broken"}), 500

# -----------------------------------------------------------------------------
# ROBUST DATA SIMULATOR GENERATOR (DEMO BACKUP)
# -----------------------------------------------------------------------------
@app.route('/demo', methods=['GET'])
def generate_demo_data():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Recreate structural tables if missing
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
            
            # Setup stable sowing date matrix
            cursor.execute("""
                INSERT INTO farm_settings (setting_key, setting_value) 
                VALUES ('sowing_date', %s) 
                ON DUPLICATE KEY UPDATE setting_value=%s
            """, ((datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d'), (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')))
            
            # Wipe historical runtime conflicts
            cursor.execute("TRUNCATE TABLE sensor_data")
            
            # Generate 14 days of clean continuous tracking telemetry rows (every 2 hours)
            start_time = datetime.now() - timedelta(days=14)
            inserted_rows = 0
            
            for hour_step in range(0, 168, 2):
                log_time = start_time + timedelta(hours=hour_step)
                # Mathematical cycles mimicking normal environment variations
                simulated_temp = float(24.0 + 6.0 * np.sin(hour_step * (np.pi / 12.0)) + np.random.normal(0, 0.5))
                simulated_hum = float(60.0 - 10.0 * np.sin(hour_step * (np.pi / 12.0)))
                simulated_soil = int(2200 + 400 * np.sin(hour_step * (np.pi / 24.0)))
                simulated_ldr = int(2000 + 1500 * np.sin(hour_step * (np.pi / 12.0)))
                simulated_dist = float(8.0 + np.random.normal(0, 0.2))
                
                cursor.execute("""
                    INSERT INTO sensor_data 
                    (timestamp, temperature, humidity, ldr_value, soil_moisture, ultrasonic_distance) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (log_time.strftime('%Y-%m-%d %H:%M:%S'), simulated_temp, simulated_hum, simulated_ldr, simulated_soil, simulated_dist))
                inserted_rows += 1
                
            connection.commit()
        return jsonify({"status": "success", "message": f"Successfully initialized a robust {inserted_rows}-row simulated data history context."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
