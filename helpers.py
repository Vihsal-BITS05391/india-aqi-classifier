# helpers.py
# Small shared file so that train_models.py and app.py use the same
# feature list and the same metric function.

import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# the 12 pollutant columns from city_day.csv plus the month taken from the Date column
FEATURES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
            "O3", "Benzene", "Toluene", "Xylene", "Month"]

TARGET = "AQI_Bucket"

# order of the categories, from cleanest air to worst air
BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def get_metrics(y_true, y_pred, y_prob):
    """Return all 6 metrics in a dict.

    The target has 6 classes so precision, recall and F1 use weighted average.
    AUC uses one vs rest on the predicted probabilities.
    """
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        # happens when some class is missing from y_true (small uploaded file for example)
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
