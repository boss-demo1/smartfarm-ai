import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("smartfarm_dataset.csv")

# =========================
# INPUT FEATURES
# =========================

X = df[[
    "temperature",
    "humidity",
    "ldr_value"
]]

# =========================
# TARGET
# =========================

y = df["light_status"]

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = RandomForestClassifier()

model.fit(X_train, y_train)

# =========================
# TEST MODEL
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "smartfarm_model.pkl")

print("Model saved!")
