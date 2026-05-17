from flask import Flask, jsonify, Response
from flask_cors import CORS
import mysql.connector
import random
import csv
import io

# =========================================
# CREATE FLASK APP
# =========================================

app = Flask(__name__)
CORS(app)

# =========================================
# DATABASE CONNECTION
# =========================================

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="smartuser",
        password="SmartFarm@123",
        database="esp"
    )

# =========================================
# HOME ROUTE
# =========================================

@app.route("/")
def home():
    return "SmartFarm Backend Running"

# =========================================
# INSERT DEMO DATA
# =========================================

@app.route("/demo")
def demo():

    db = get_db()
    cursor = db.cursor(dictionary=True)

    temperature = round(random.uniform(20, 35), 2)
    humidity = round(random.uniform(40, 80), 2)
    ldr_value = random.randint(0, 4095)

    light_status = 1 if ldr_value < 1500 else 0

    sql = """
    INSERT INTO sensor_data
    (temperature, humidity, ldr_value, light_status)
    VALUES (%s, %s, %s, %s)
    """

    cursor.execute(sql, (
        temperature,
        humidity,
        ldr_value,
        light_status
    ))

    db.commit()
    db.close()

    return jsonify({
        "status": "inserted",
        "temperature": temperature,
        "humidity": humidity,
        "ldr_value": ldr_value,
        "light_status": light_status
    })

# =========================================
# GET LATEST DATA
# =========================================

@app.route("/latest")
def latest():

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1"
    )

    data = cursor.fetchone()

    db.close()

    return jsonify(data)

# =========================================
# GET HISTORY
# =========================================

@app.route("/sensor-data")
def sensor_data():

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM sensor_data ORDER BY id DESC LIMIT 100"
    )

    data = cursor.fetchall()

    db.close()

    return jsonify(data)

# =========================================
# CHART DATA
# =========================================

@app.route("/chart-data")
def chart_data():

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM sensor_data ORDER BY id DESC LIMIT 50"
    )

    data = cursor.fetchall()

    db.close()

    return jsonify(data)

# =========================================
# EXPORT CSV
# =========================================

@app.route("/export/csv")
def export_csv():

    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM sensor_data")
    rows = cursor.fetchall()

    db.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "id",
        "timestamp",
        "temperature",
        "humidity",
        "ldr_value",
        "light_status"
    ])

    for r in rows:
        writer.writerow([
            r["id"],
            r["timestamp"],
            r["temperature"],
            r["humidity"],
            r["ldr_value"],
            r["light_status"]
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=sensor_data.csv"
        }
    )

# =========================================
# RUN SERVER
# =========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
