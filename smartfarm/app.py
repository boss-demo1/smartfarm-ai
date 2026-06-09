import os
import random
from datetime import datetime
import pymysql
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained machine learning model binaries on startup
try:
    irrigation_model = joblib.load('irrigation_model.pkl')
    irrigation_scaler = joblib.load('irrigation_scaler.pkl')
    yield_model = joblib.load('yield_model.pkl')
    print("🚀 All ML Models and Scalers loaded successfully into memory!")
except Exception as e:
    print(f"⚠️ Error loading ML model binaries: {e}")

# Database connection helper
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
# NEW DIAGNOSTIC HEALTH ROUTE (Addresses your /health link request)
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
    
    # 1. Verify Machine Learning Binaries
    if 'irrigation_model' in globals() and 'yield_model' in globals():
        health_status["models_loaded"] = True
        
    # 2. Verify Database Connection
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
# PHASE 1, TASK 1.2: DYNAMIC LOGIC FOR SOWING DATE & PLANT AGE
# -----------------------------------------------------------------------------
def get_plant_age_days():
    """Queries farm_settings to compute how many days the crop has been alive."""
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Check if table exists first before running query
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = DATABASE() AND table_name = 'farm_settings'
            """)
            if cursor.fetchone()['COUNT(*)'] == 0:
                return 0.0

            sql = "SELECT setting_value FROM farm_settings WHERE setting_key = 'sowing_date'"
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if result:
                sowing_date_str = result['setting_value']
                sowing_date = datetime.strptime(sowing_date_str, '%Y-%m-%d')
                current_date = datetime.now()
                delta = current_date - sowing_date
                return max(0.0, delta.total_seconds() / 86400.0)
            return 0.0
    except Exception as e:
        print(f"Error reading dynamic farm_settings: {e}")
        return 0.0
    finally:
        if 'connection' in locals():
            connection.close()

# -----------------------------------------------------------------------------
# MACHINE LEARNING ENGINE FEATURE CALCULATOR
# -----------------------------------------------------------------------------
def calculate_live_metrics():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Check if tables exist
            cursor.execute("SHOW TABLES LIKE 'farm_settings'")
            if not cursor.fetchone():
                return 0.0, 0.0, 0.0, 0.0
                
            cursor.execute("SELECT setting_value FROM farm_settings WHERE setting_key = 'sowing_date'")
            sowing_res = cursor.fetchone()
            if not sowing_res:
                return 0.0, 0.0, 0.0, 0.0
            
            sowing_date = datetime.strptime(sowing_res['setting_value'], '%Y-%m-%d')
            current_time = datetime.now()
            day_number = max(0.0, (current_time - sowing_date).total_seconds() / 86400.0)
            
            cursor.execute("SHOW TABLES LIKE 'sensor_data'")
            if not cursor.fetchone():
                return day_number, day_number * 12.5, 0.0, day_number * 15.0

            cursor.execute("SELECT timestamp, temperature, soil_moisture FROM sensor_data ORDER BY timestamp ASC")
            logs = cursor.fetchall()
            
            if not logs:
                approx_gdd = day_number * 12.5
                return day_number, approx_gdd, 0.0, day_number * 15.0
            
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date_only'] = df['timestamp'].dt.date
            daily_groups = df.groupby('date_only')
            
            cumulative_gdd = 0.0
            for date_item, group in daily_groups:
                daily_avg = (group['temperature'].max() + group['temperature'].min()) / 2.0
                gdd_today = max(0.0, daily_avg - 10.0)
                cumulative_gdd += gdd_today
                
            df['time_delta_hours'] = df['timestamp'].diff().dt.total_seconds().fillna(15.0) / 3600.0
            cumulative_heat_stress = (df.loc[df['temperature'] > 30.0, 'time_delta_hours'].sum())
            cumulative_water = ((4095 - df['soil_moisture']) * df['time_delta_hours']).sum() / 1000.0
            
            return day_number, cumulative_gdd, cumulative_heat_stress, cumulative_water
            
    except Exception as e:
        print(f"Error calculating live cumulative features: {e}")
        return 0.0, 0.0, 0.0, 0.0
    finally:
        if 'connection' in locals():
            connection.close()

# -----------------------------------------------------------------------------
# PRODUCTION PREDICT ENDPOINT PIPELINE
# -----------------------------------------------------------------------------
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # 1. Capture your exact hardware sensor readings from incoming request arguments
        temp = float(request.args.get('temperature', 25.0))
        hum = float(request.args.get('humidity', 50.0))
        ldr = int(request.args.get('ldr_value', 2000))
        soil = int(request.args.get('soil_moisture', 2000))
        ultrasonic = float(request.args.get('ultrasonic_distance', 5.0)) # default 5cm (full)
        
        # Pull timeline parameters and cumulative weather history summaries
        day_number, cum_gdd, heat_stress, cum_water = calculate_live_metrics()
        
        # 2. Machine Learning Pipeline: Adjusting for Evaporation & Crop Timeline
        input_data_irr = [[temp, hum, soil, day_number]]
        scaled_features_irr = irrigation_scaler.transform(input_data_irr)
        ml_predicted_duration = float(irrigation_model.predict(scaled_features_irr)[0])
        
        # 3. Apply your explicit rule-based override logic
        pump1_status = 1 if ultrasonic > 10.0 else 0
        pump2_status = 1 if soil > 2500 else 0
        relay4_light = 1 if ldr <= 300 else 0
        
        # Yield Forecasting ML calculations
        input_data_yield = [[day_number, cum_gdd, heat_stress, cum_water]]
        yield_preds = yield_model.predict(input_data_yield)[0]
        
        if day_number <= 15:
            growth_stage = "Germination / Emergence"
        elif day_number <= 45:
            growth_stage = "Vegetative Development"
        else:
            growth_stage = "Flowering / Maturation"

        # 4. Construct response payload matching your exact hardware map
        response_data = {
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
                "servo_schedule": "Every 20s rotate 0 to 180 for 3s"
            },
            "agronomic_yield_forecast": {
                "current_growth_stage": growth_stage,
                "accumulated_gdd": round(cum_gdd, 1),
                "expected_yield_grams": round(max(0.0, yield_preds[0]), 1),
                "projected_days_until_harvest": max(0, int(110 - day_number))
            }
        }
        return jsonify(response_data)
        
    except Exception as err:
        return jsonify({"error": str(err)}), 500
# -----------------------------------------------------------------------------
# RESTORED ROUTE 1: Fetch the single latest sensor log entry for the dashboard
# -----------------------------------------------------------------------------
@app.route('/latest', methods=['GET'])
def get_latest():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'sensor_data'")
            if not cursor.fetchone():
                return jsonify({"error": "Table 'sensor_data' does not exist"}), 200

            sql = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1"
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if result and isinstance(result, dict):
                if 'timestamp' in result and hasattr(result['timestamp'], 'strftime'):
                    result['timestamp'] = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                return jsonify(result)
            return jsonify({"status": "waiting for logs", "message": "No sensor rows found"}), 200
    except Exception as e:
        return jsonify({"error": "DB Processing Error", "details": str(e)}), 500
    finally:
        if 'connection' in locals():
            connection.close()

# -----------------------------------------------------------------------------
# RESTORED ROUTE 2: Fetch historical records for telemetry charts (e.g., limit=50)
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
# NEW AUTOMATED DEMO DATA GENERATOR ROUTE
# -----------------------------------------------------------------------------
@app.route('/demo', methods=['GET'])
def generate_demo_data():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 1. Generate realistic, randomized smart farm sensor metrics
            temperature = round(random.uniform(18.0, 38.0), 1)      # 18°C to 38°C
            humidity = round(random.uniform(30.0, 85.0), 1)         # 30% to 85%
            ldr_value = random.randint(100, 4095)                   # Spans above/below your 300 threshold
            soil_moisture = random.randint(1000, 4500)              # Spans above/below your 2500 threshold
            ultrasonic_distance = round(random.uniform(2.0, 20.0), 1) # 2cm to 20cm (crosses your 10cm threshold)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 2. Insert the randomized row into your live Railway database
            sql = """
                INSERT INTO sensor_data 
                (timestamp, temperature, humidity, ldr_value, soil_moisture, ultrasonic_distance) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (timestamp, temperature, humidity, ldr_value, soil_moisture, ultrasonic_distance))
            connection.commit()

            # 3. Formulate the real-time hardware status simulation response
            response = {
                "status": "success",
                "message": "Successfully generated and logged simulated sensor hardware metrics!",
                "data_logged": {
                    "timestamp": timestamp,
                    "environmental_sensors": {
                        "temperature_celsius": temperature,
                        "humidity_percentage": humidity,
                        "ldr_ambient_light": ldr_value,
                        "soil_moisture_raw": soil_moisture,
                        "ultrasonic_tank_distance_cm": ultrasonic_distance
                    },
                    "simulated_hardware_actions": {
                        "pump1_tank_refill": "ON" if ultrasonic_distance > 10.0 else "OFF",
                        "pump2_irrigation": "ON" if soil_moisture > 2500 else "OFF",
                        "relay4_growth_light": "ON" if ldr_value <= 300 else "OFF",
                        "servo_motor": "Active structural 20s interval cycle"
                    }
                }
            }
            return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        connection.close()



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
