import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("smartfarm_dataset.csv")

# =====================================
# INPUT FEATURES
# =====================================

X = df[[
    "temperature",
    "humidity",
    "ldr_value"
]]

# =====================================
# TARGET
# =====================================

y = df["light_status"]

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# CREATE MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

model.fit(X_train, y_train)

# =====================================
# PREDICTIONS
# =====================================

predictions = model.predict(X_test)

# =====================================
# ACCURACY
# =====================================

accuracy = accuracy_score(y_test, predictions)

print("\\nModel Accuracy:")
print(accuracy)

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\\nClassification Report:")
print(classification_report(y_test, predictions))

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, predictions)

print("\\nConfusion Matrix:")
print(cm)

# =====================================
# DISPLAY MATRIX
# =====================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.show()

# =====================================
# FEATURE IMPORTANCE
# =====================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

plt.bar(features, importance)

plt.title("Feature Importance")

plt.ylabel("Importance")

plt.show()
