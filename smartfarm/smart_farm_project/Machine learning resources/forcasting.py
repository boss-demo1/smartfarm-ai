"""
predictors.py
-------------
Loads the trained scikit-learn models from disk (once, on first call)
and exposes two clean prediction functions used by flaa.py:

  predict_irrigation(...)  →  pump seconds  (float)
  predict_yield(...)       →  dict with grams, pods, growth_stage, harvest_date
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────
# Lazy-loaded model cache  (load once, reuse forever)
# ─────────────────────────────────────────────────────────────

_irr_model   = None   # RandomForestRegressor
_irr_scaler  = None   # StandardScaler for irrigation
_yld_model   = None   # GradientBoostingRegressor
_yld_scaler  = None   # StandardScaler for yield


def _load_irrigation(model_dir: str):
    """Load irrigation model + scaler from model_dir if not already loaded."""
    global _irr_model, _irr_scaler
    if _irr_model is None:
        # CORRECT — matches what irrigation_of_plant.py saved
        _irr_model = joblib.load(os.path.join(model_dir, "irrigation_model_last2.pkl"))
        _irr_scaler = joblib.load(os.path.join(model_dir, "irrigation_scaler_last2.pkl"))
        print("[PREDICTOR] Irrigation model loaded")


def _load_yield(model_dir: str):
    """Load yield model + scaler from model_dir if not already loaded."""
    global _yld_model, _yld_scaler
    if _yld_model is None:
        _yld_model  = joblib.load(os.path.join(model_dir, "yield_model_last2.pkl"))
        _yld_scaler = joblib.load(os.path.join(model_dir, "yield_scaler_last2.pkl"))
        print("[PREDICTOR] Yield model loaded")


# ─────────────────────────────────────────────────────────────
# GDD thresholds (must match yeald_of_plant.py)
# ─────────────────────────────────────────────────────────────

GDD_FLOWERING = 850
GDD_POD_FILL  = 1000
GDD_MATURITY  = 1450


def _growth_stage(cumulative_gdd: float) -> str:
    """Return the current growth stage name based on accumulated GDD."""
    if cumulative_gdd < GDD_FLOWERING:
        return "Vegetative"
    elif cumulative_gdd < GDD_POD_FILL:
        return "Flowering"
    elif cumulative_gdd < GDD_MATURITY:
        return "Pod Fill"
    else:
        return "Mature"


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def predict_irrigation(
    soil: float,
    temp: float,
    humidity: float,
    light: float,
    water_level: float,
    time_day: int,
    model_dir: str
) -> float:
    """
    Predict how many seconds the irrigation pump should run.

    Parameters match the features used in irrigation_of_plant.py:
      soil, temp, humidity, light, water_level, time_day

    Returns
    -------
    float  seconds to run pump (0 – 30)
    """
    _load_irrigation(model_dir)

    # Build a single-row DataFrame with the exact column order used during training
    X = pd.DataFrame([{
        "soil":        soil,
        "temp":        temp,
        "humidity":    humidity,
        "light":       light,
        "water_level": water_level,
        "time_day":    time_day,
    }])

    X_sc  = _irr_scaler.transform(X)
    pump  = float(_irr_model.predict(X_sc)[0])
    pump  = float(np.clip(pump, 0.0, 30.0))

    return round(pump, 1)


def predict_yield(
    cumulative_gdd: float,
    cumulative_water: float,
    cumulative_temp_stress: float,
    cumulative_humidity_stress: int,
    cumulative_light: float,
    day_number: int,
    temp: float,
    humidity: float,
    soil: float,
    light: float,
    water_level: float,
    model_dir: str
) -> dict:
    """
    Predict expected yield (grams + pods) and estimated harvest date.

    Parameters match the YIELD_FEATURES list in yeald_of_plant.py.

    Returns
    -------
    dict with keys:
        growth_stage    (str)
        pods            (float)
        grams           (float)
        harvest_date    (str  YYYY-MM-DD)
        days_remaining  (int)
    """
    _load_yield(model_dir)

    stage = _growth_stage(cumulative_gdd)

    # Build feature row — ORDER must match YIELD_FEATURES in yeald_of_plant.py
    X = pd.DataFrame([{
        "cumulative_gdd":               cumulative_gdd,
        "cumulative_water":             cumulative_water,
        "cumulative_temp_stress":       cumulative_temp_stress,
        "cumulative_humidity_stress":   cumulative_humidity_stress,
        "cumulative_light":             cumulative_light,
        "temp":                         temp,
        "humidity":                     humidity,
        "soil":                         soil,
        "light":                        light,
        "water_level":                  water_level,
        "day_number":                   day_number,
    }])

    X_sc  = _yld_scaler.transform(X)
    grams = float(np.clip(_yld_model.predict(X_sc)[0], 10.0, 30.0))
    pods  = round(grams / 0.6, 1)

    # Estimate days until harvest based on average GDD per day
    avg_gdd_per_day = cumulative_gdd / max(day_number, 1)
    remaining_gdd   = max(0.0, GDD_MATURITY - cumulative_gdd)

    if avg_gdd_per_day > 0:
        days_remaining = int(np.ceil(remaining_gdd / avg_gdd_per_day))
    else:
        days_remaining = 999

    harvest_date = date.today() + timedelta(days=days_remaining)

    return {
        "growth_stage":   stage,
        "pods":           pods,
        "grams":          round(grams, 2),
        "harvest_date":   str(harvest_date),
        "days_remaining": days_remaining,
    }
