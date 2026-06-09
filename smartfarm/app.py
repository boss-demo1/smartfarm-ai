import os
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
        host=os.getenv('MYSQLHOST', 'mysql.railway.internal'),
        user=os.getenv('MYSQLUSER', 'root'),
        password=os.getenv('MYSQLPASSWORD', 'rQjNEHvAHmbqgFfnvRZnZRpPcEnBctqd'),
        database=os.getenv('MYSQLDATABASE', 'railway'),
        port=int(os.getenv('MYSQLPORT', 3306)),
        cursorclass=pymysql.cursors.DictCursor
    )

# -----------------------------------------------------------------------------
# PHASE 1, TASK 1.2: DYNAMIC LOGIC FOR SOWING DATE & PLANT AGE
# -----------------------------------------------------------------------------
def get_plant_age_days():
    """Queries farm_settings to compute how many days the crop has been alive."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT setting_value FROM farm_settings WHERE setting_key = 'sowing_date'"
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if result:
                sowing_date_str = result['setting_value']  # '2026-05-22'
                sowing_date = datetime.strptime(sowing_date_str, '%Y-%m-%d')
                current_date = datetime.now()
                
                # Calculate the exact difference in total decimal days
                delta = current_date - sowing_date
                day_number = delta.total_seconds() / 86400.0
                return max(0.0, day_number) # Prevent negative numbers
            else:
                return 0.0
    except Exception as e:
        print(f"Error reading dynamic farm_settings: {e}")
        return 0.0
    finally:
        connection.close()

# -----------------------------------------------------------------------------
# MACHINE LEARNING ENGINE FEATURE CALCULATOR
# -----------------------------------------------------------------------------
def calculate_live_metrics():
    """
    Queries history from the database to compute real-time cumulative features
    based on the actual sowing date.
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 1. Fetch Sowing Date
            cursor.execute("SELECT setting_value FROM farm_settings WHERE setting_key = 'sowing_date'")
            sowing_res = cursor.fetchone()
            if not sowing_res:
                return 0.0, 0.0, 0.0, 0.0
            
            sowing_date = datetime.strptime(sowing_res['setting_value'], '%Y-%m-%d')
            current_time = datetime.now()
            
            # Calculate fractional day number
            day_number = max(0.0, (current_time - sowing_date).total_seconds() / 86400.0)
            
            # 2. Fetch recent sensor logs to build cumulative approximations
            cursor.execute("SELECT timestamp, temperature, soil_moisture FROM sensor_data ORDER BY timestamp ASC")
            logs = cursor.fetchall()
            
            if not logs:
                # Fallback default values based on age if table is clear
                approx_gdd = day_number * 12.5
                return day_number, approx_gdd, 0.0, day_number * 15.0
            
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Simple aggregations
            df['date_only'] = df['timestamp'].dt.date
            daily_groups = df.groupby('date_only')
            
            cumulative_gdd = 0.0
            for date_item, group in daily_groups:
                daily_avg = (group['temperature'].max() + group['temperature'].min()) / 2.0
                gdd_today = max(0.0, daily_avg - 10.0)
                cumulative_gdd += gdd_today
                
            # Time differences for stress and water integrals
            df['time_delta_hours'] = df['timestamp'].diff().dt.total_seconds().fillna(15.0) / 3600.0
            cumulative_heat_stress = (df.loc[df['temperature'] > 30.0, 'time_delta_hours'].sum())
            cumulative_water = ((4095 - df['soil_moisture']) * df['time_delta_hours']).sum() / 1000.0
            
            return day_number, cumulative_gdd, cumulative_heat_stress, cumulative_water
            
    except Exception as e:
        print(f"Error calculating live cumulative features: {e}")
        return 0.0, 0.0, 0.0, 0.0
    finally:
        connection.close()

# -----------------------------------------------------------------------------
# PRODUCTION PREDICT ENDPOINT PIPELINE
# -----------------------------------------------------------------------------
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get live sensor arguments from request parameters
        temp = float(request.args.get('temperature', 25.0))
        hum = float(request.args.get('humidity', 50.0))
        ldr = int(request.args.get('ldr_value', 2000))
        soil = int(request.args.get('soil_moisture', 2000))
        
        # Pull our timeline parameters and cumulative sums
        day_number, cum_gdd, heat_stress, cum_water = calculate_live_metrics()
        
        # 1. Compute Active Irrigation Duration Prediction
        input_data_irr = [[temp, hum, soil, day_number]]
        scaled_features_irr = irrigation_scaler.transform(input_data_irr)
        predicted_duration = irrigation_model.predict(scaled_features_irr)[0]
        
        # 2. Compute Yield Forecast Prediction
        input_data_yield = [[day_number, cum_gdd, heat_stress, cum_water]]
        yield_preds = yield_model.predict(input_data_yield)[0]
        predicted_grams = yield_preds[0]
        predicted_pods = int(yield_preds[1])
        
        # Classify the biological development growth stage window
        if day_number <= 15:
            growth_stage = "Germination / Emergence"
        elif day_number <= 45:
            growth_stage = "Vegetative Development"
        elif day_number <= 75:
            growth_stage = "Flowering & Pod Setting"
        else:
            growth_stage = "Maturation / Senescence"

        light_decision = 1 if ldr > 3000 else 0

        # Construct JSON output array to match index.html properties exactly
        response_data = {
            "actuator_predictions": {
                "relay2_predicted_duration_seconds": max(0.0, float(predicted_duration)),
                "relay3_growth_light": light_decision
            },
            "agronomic_yield_forecast": {
                "current_growth_stage": growth_stage,
                "accumulated_gdd": round(cum_gdd, 1),
                "estimated_harvest_date": "2026-09-10", 
                "projected_days_until_harvest": max(0, int(110 - day_number)),
                "expected_yield_grams": round(max(0.0, predicted_grams), 1),
                "expected_yield_pods": max(0, predicted_pods)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as err:
        return jsonify({"error": str(err), "details": "Verify your model .pkl binaries exist in the root folder"}), 500

# -----------------------------------------------------------------------------
# RESTORED ROUTE 1: Fetch the single latest sensor log entry for the dashboard
# -----------------------------------------------------------------------------
@app.route('/latest', methods=['GET'])
def get_latest():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1"
            cursor.execute(sql)
            result = cursor.fetchone()
            if result:
                if 'timestamp' in result and isinstance(result['timestamp'], datetime):
                    result['timestamp'] = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                return jsonify(result)
            return jsonify({"error": "No sensor data logs found in database"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

# -----------------------------------------------------------------------------
# RESTORED ROUTE 2: Fetch historical records for telemetry charts (e.g., limit=50)
# -----------------------------------------------------------------------------
@app.route('/sensor-data', methods=['GET'])
def get_sensor_data():
    limit = request.args.get('limit', default=50, type=int)
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT %s"
            cursor.execute(sql, (limit,))
            results = cursor.fetchall()
            
            for row in results:
                if 'timestamp' in row and isinstance(row['timestamp'], datetime):
                    row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
