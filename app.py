# app.py
# Streamlit app for the Air Quality India classification assignment.
# Run with:  streamlit run app.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from helpers import FEATURES, TARGET, get_metrics

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Air Quality India Classifier", layout="wide")
st.title("AQI Category Prediction for Indian Cities")
st.write("Predicts the AQI bucket (Good to Severe) from pollutant levels. "
         "Dataset: Air Quality Data in India, 2015 to 2020 (Kaggle).")


# cache so the model files are not read again on every click
@st.cache_resource
def load_model(file_name):
    return joblib.load(os.path.join(MODEL_DIR, file_name))


@st.cache_resource
def load_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))


# stop early with a clear message if the model files are missing
if not os.path.exists(os.path.join(MODEL_DIR, "label_encoder.joblib")):
    st.error("Model files not found. Run train_models.py first.")
    st.stop()

encoder = load_encoder()

# sidebar
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
st.sidebar.caption("The CSV needs these columns: " + ", ".join(FEATURES) +
                   ". Add " + TARGET + " too if you want the metrics.")

model = load_model(MODEL_FILES[model_name])

tab1, tab2, tab3 = st.tabs(["Evaluate on test data", "Try one reading", "Model comparison"])

# tab 1: upload a csv and see metrics
with tab1:
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif os.path.exists("test_data.csv"):
        st.info("No file uploaded, so the default test_data.csv is being used.")
        data = pd.read_csv("test_data.csv")
    else:
        data = None

    if data is None:
        st.warning("Please upload a CSV file from the sidebar.")
    else:
        missing_cols = [c for c in FEATURES if c not in data.columns]
        if missing_cols:
            st.error("These columns are missing in the file: " + ", ".join(missing_cols))
        else:
            st.write("Rows in file:", len(data))
            X = data[FEATURES]
            pred = model.predict(X)
            prob = model.predict_proba(X)
            pred_text = encoder.inverse_transform(pred)

            if TARGET in data.columns:
                y_true = encoder.transform(data[TARGET])
                scores = get_metrics(y_true, pred, prob)

                st.subheader("Metrics for " + model_name)
                cols = st.columns(6)
                for col, (k, v) in zip(cols, scores.items()):
                    col.metric(k, "nan" if np.isnan(v) else round(v, 4))

                left, right = st.columns(2)
                with left:
                    st.subheader("Confusion matrix")
                    cm = confusion_matrix(y_true, pred, labels=range(len(encoder.classes_)))
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                with right:
                    st.subheader("Classification report")
                    report = classification_report(
                        y_true, pred, labels=range(len(encoder.classes_)),
                        target_names=encoder.classes_, output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).transpose().round(3))
            else:
                st.info(TARGET + " column not found, so only predictions are shown.")

            out = data.copy()
            out["Predicted_" + TARGET] = pred_text
            st.subheader("Predictions")
            st.dataframe(out.head(100))

# tab 2: single reading
with tab2:
    st.write("Enter pollutant values (in micrograms per cubic metre, CO in mg per cubic metre).")
    defaults = {"PM2.5": 80.0, "PM10": 150.0, "NO": 15.0, "NO2": 35.0, "NOx": 40.0,
                "NH3": 20.0, "CO": 1.2, "SO2": 12.0, "O3": 35.0, "Benzene": 2.0,
                "Toluene": 5.0, "Xylene": 1.5}
    values = {}
    cols = st.columns(4)
    for i, f in enumerate(FEATURES):
        with cols[i % 4]:
            if f == "Month":
                values[f] = st.number_input("Month (1 to 12)", 1, 12, 11)
            else:
                values[f] = st.number_input(f, min_value=0.0, value=defaults[f])

    if st.button("Predict AQI category"):
        row = pd.DataFrame([values])[FEATURES]
        p = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        st.success("Predicted category: " + encoder.inverse_transform([p])[0])
        st.bar_chart(pd.Series(probs, index=encoder.classes_))

# tab 3: all five models on the same data
# runs only predict on the saved models, nothing is retrained here.
# cached so it does not run again on every click in the sidebar
@st.cache_data
def compare_all_models(df):
    y_all = encoder.transform(df[TARGET])
    rows = []
    for m_name, m_file in MODEL_FILES.items():
        m = load_model(m_file)
        m_pred = m.predict(df[FEATURES])
        m_prob = m.predict_proba(df[FEATURES])
        sc = get_metrics(y_all, m_pred, m_prob)
        sc["ML Model Name"] = m_name
        rows.append(sc)
    return pd.DataFrame(rows).set_index("ML Model Name").round(4)


with tab3:
    st.write("All five saved models run on the data from the first tab.")
    if data is not None and all(c in data.columns for c in FEATURES + [TARGET]):
        comp = compare_all_models(data)
        st.dataframe(comp)
        st.caption("Best model by MCC on this data: " + comp["MCC"].idxmax())
    else:
        st.warning("Upload a CSV with all feature columns and the " + TARGET +
                   " column to compare the models.")

    # the table that was saved when train_models.py was run
    path = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(path):
        st.subheader("Saved results from the training run")
        st.dataframe(pd.read_csv(path))
