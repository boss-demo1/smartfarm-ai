"""
SmartFarm IoT v3.0 — ML Training Script
Run once locally to generate smartfarm_model.pkl
Then push to GitHub so Render loads it on startup.

Usage:
    cd ~/smartfarm
    python3 train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

CSV_PATH   = os.path.join(os.path.dirname(__file__), "smartfarm_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "smartfarm_model.pkl")

print("=" * 55)
print("  SmartFarm v3.0 — ML Training")
print("=" * 55)

# ── Load CSV ──────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✓ Loaded {len(df)} rows from CSV")
except FileNotFoundError:
    print(f"✗ ERROR: {CSV_PATH} not found")
    exit(1)

# ── Add missing columns with defaults ─────────────────────────
# The CSV was generated with 3 sensors — we pad the new ones
if "soil_moisture" not in df.columns:
    import random
    df["soil_moisture"] = [random.randint(1000, 4000)
                           for _ in range(len(df))]
    print("  Added soil_moisture column (random defaults)")

if "distance" not in df.columns:
    import random
    df["distance"] = [round(random.uniform(5, 40), 2)
                      for _ in range(len(df))]
    print("  Added distance column (random defaults)")

# Handle both column name versions
if "light_status" in df.columns and "relay3_light" not in df.columns:
    df["relay3_light"] = df["light_status"]

# ── Clean ─────────────────────────────────────────────────────
FEATURES = ["temperature", "humidity", "ldr_value",
            "soil_moisture", "distance"]
TARGET   = "relay3_light"

df = df[FEATURES + [TARGET]].dropna()
print(f"✓ Clean rows: {len(df)}")
print(f"\n  relay3_light distribution:")
print(f"  OFF (0): {(df[TARGET] == 0).sum()} rows")
print(f"  ON  (1): {(df[TARGET] == 1).sum()} rows")

# ── Split ─────────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n  Train: {len(X_train)} rows")
print(f"  Test : {len(X_test)} rows")

# ── Train ─────────────────────────────────────────────────────
print("\n  Training Random Forest (100 trees)...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Accuracy : {round(accuracy * 100, 2)} %")
print(f"\n{classification_report(y_test, y_pred, target_names=['Light OFF','Light ON'])}")

# ── Feature importance ────────────────────────────────────────
print("  Feature Importance:")
for name, score in zip(FEATURES, model.feature_importances_):
    bar = "█" * int(score * 40)
    print(f"  {name:<15} {bar} {round(score * 100, 1)}%")

# ── Save ──────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"\n✓ Model saved → {MODEL_PATH}")

# ── Quick test ────────────────────────────────────────────────
print("\n  Quick prediction tests:")
tests = [
    {"temperature": 28.0, "humidity": 65.0, "ldr_value": 3900,
     "soil_moisture": 1800, "distance": 20.0},
    {"temperature": 23.0, "humidity": 72.0, "ldr_value": 400,
     "soil_moisture": 3100, "distance":  7.0},
    {"temperature": 31.0, "humidity": 55.0, "ldr_value": 2000,
     "soil_moisture": 2000, "distance": 15.0},
]
for t in tests:
    pred = int(model.predict(pd.DataFrame([t]))[0])
    print(f"  ldr={t['ldr_value']} soil={t['soil_moisture']} "
          f"dist={t['distance']}cm → Light {'ON' if pred else 'OFF'}")

print("\n  Next steps:")
print("  git add smartfarm_model.pkl")
print("  git commit -m 'v3 retrained model 5 features'")
print("  git push origin render-deploy")
print("=" * 55)
