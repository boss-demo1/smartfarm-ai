import pandas as pd
import joblib

# Load trained model
model = joblib.load("smartfarm_model.pkl")

# Create input data with feature names
data = pd.DataFrame([{
    "temperature": 28,
    "humidity": 60,
    "ldr_value": 500
}])

# Predict
prediction = model.predict(data)

print("Prediction:", prediction)
