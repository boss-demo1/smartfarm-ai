import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("smartfarm_dataset.csv")

# Add missing columns with default values if not present
if "soil_moisture" not in df.columns:
    df["soil_moisture"] = 1500
if "distance" not in df.columns:
    df["distance"] = 15.0

# Rename light_status to relay4_light if needed
if "light_status" in df.columns and "relay4_light" not in df.columns:
    df["relay4_light"] = df["light_status"]

df = df[["temperature", "humidity", "ldr_value",
         "soil_moisture", "distance", "relay4_light"]].dropna()

X = df[["temperature", "humidity", "ldr_value",
        "soil_moisture", "distance"]]
y = df["relay4_light"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100,
                                random_state=42,
                                max_depth=10)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {round(accuracy * 100, 2)} %")

joblib.dump(model, "smartfarm_model.pkl")
print("Model saved as smartfarm_model.pkl")
