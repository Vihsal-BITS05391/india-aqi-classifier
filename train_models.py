# train_models.py
# ML Assignment 2: classification on "Air Quality Data in India (2015 to 2020)"
# Target is AQI_Bucket (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)
#
# How to run:  python train_models.py
# Put city_day.csv from Kaggle in the same folder as this file first.

import os
import sys
import warnings
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # so it also works on a machine without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from helpers import FEATURES, TARGET, BUCKET_ORDER, get_metrics

# newer sklearn versions print a future warning for SVC(probability=True), hiding it
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_FILE = "city_day.csv"
MODEL_DIR = "model"
RANDOM_STATE = 42

if len(sys.argv) > 1:
    DATA_FILE = sys.argv[1]

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. load the data
df = pd.read_csv(DATA_FILE)
print("Raw shape:", df.shape)

# rows without a bucket cannot be used for training, so remove them
df = df.dropna(subset=[TARGET])
print("Shape after removing rows with no AQI_Bucket:", df.shape)

# AQI column is NOT used as a feature. The bucket is made directly from AQI,
# so keeping it would be cheating (the model would just read the answer).
# Month is taken from Date because pollution is clearly worse in winter.
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

X = df[FEATURES]
y_text = df[TARGET]

# 2. encode target labels as numbers, in the order of severity
encoder = LabelEncoder()
encoder.fit(BUCKET_ORDER)
y = encoder.transform(y_text)
print("\nClass counts:")
print(y_text.value_counts())

# 3. train test split (stratify so every class appears in both parts)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    # probability=True is needed for the AUC score, it makes SVM slower to train
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

results = []

for name, clf in models.items():
    print("\nTraining", name, "...")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clf),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    scores = get_metrics(y_test, y_pred, y_prob)
    scores["ML Model Name"] = name
    results.append(scores)
    print({k: round(v, 4) for k, v in scores.items() if k != "ML Model Name"})

    # save the trained pipeline, file name like logistic_regression.joblib
    file_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, file_name))

    # confusion matrix picture for the report
    cm = confusion_matrix(y_test, y_pred, labels=range(len(BUCKET_ORDER)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Confusion matrix: " + name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "cm_" + file_name.replace(".joblib", ".png")), dpi=120)
    plt.close()

# 5. save the comparison table, the label encoder and the test file
table = pd.DataFrame(results)[["ML Model Name", "Accuracy", "AUC", "Precision",
                               "Recall", "F1", "MCC"]]
table = table.round(4)
table.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

# test_data.csv is the untouched 20 percent. It has the original text label,
# so it can be uploaded directly in the streamlit app.
test_out = X_test.copy()
test_out[TARGET] = encoder.inverse_transform(y_test)
test_out.to_csv("test_data.csv", index=False)

print("\n===== Comparison table =====")
print(table.to_string(index=False))