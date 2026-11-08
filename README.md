# AQI Category Prediction for Indian Cities

ML Assignment 2: classification on the Air Quality Data in India (2015 to 2020) dataset.

## a. Problem statement

Air pollution is a serious public health problem in many Indian cities, especially Delhi, Patna, Lucknow and Ahmedabad. The official Air Quality Index (AQI) is shown to the public as a category such as Good, Poor or Severe. The aim of this project is to predict the AQI category of a city on a given day using only the measured pollutant levels and the month. Five classification models are trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and MCC.

A model like this can help in early health alerts when some sensors are missing or the AQI value itself has not been published yet.

## b. Dataset description

- **Source:** Kaggle, "Air Quality Data in India (2015 - 2020)" by Rohan Rao. File used: `city_day.csv`.
- **Raw size:** 29,531 daily records from 26 Indian cities. After removing rows with no `AQI_Bucket`, 24,850 rows were left for modelling.
- **Type:** multi-class classification with 6 classes.
- **Class counts after cleaning:** Moderate 8829, Satisfactory 8224, Poor 2781, Very Poor 2337, Good 1341, Severe 1338. The classes are imbalanced, with Moderate and Satisfactory making up about two thirds of the data.
- **Target:** `AQI_Bucket` with values Good, Satisfactory, Moderate, Poor, Very Poor and Severe.
- **Features (13):** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene and Month (taken from the Date column).

Preprocessing steps:

1. Rows with no `AQI_Bucket` were dropped because they have no label.
2. The `AQI` column was removed from the features. The bucket is calculated directly from it, so keeping it would be target leakage.
3. Missing pollutant values (a lot of them: Xylene is 61 percent missing, PM10 38 percent and NH3 35 percent) were filled with the median, learned from the training set only.
4. Features were standardised using StandardScaler.
5. 80 / 20 stratified train test split with `random_state = 42` (19,880 training rows and 4,970 test rows).

Imputer and scaler are inside a scikit-learn Pipeline together with each model, so the saved `.joblib` files can take raw data directly.

## c. GitHub repository link

https://github.com/Vihsal-BITS05391/india-aqi-classifier

Streamlit app: PASTE_STREAMLIT_LINK_HERE

Repository structure:

```
project-folder/
|   app.py
|   train_models.py
|   explore.py
|   helpers.py
|   .gitignore
|   requirements.txt
|   README.md
|   test_data.csv
|	eda_plots/
|		class_balance.png
|		correlation.png
|		missing_values.png
|		pm25_by_month.png
|   model/
|       logistic_regression.joblib
|       decision_tree.joblib
|       knn.joblib
|       naive_bayes.joblib
|       support_vector_machine.joblib
|       label_encoder.joblib
|       metrics.csv
|       cm_*.png (confusion matrix of each model)
```

How to run:

1. Download `city_day.csv` from Kaggle into the project folder.
2. `pip install -r requirements.txt`
3. `python explore.py` (optional, first look at the data and plots)
4. `python train_models.py` (trains all 5 models, saves them and creates `test_data.csv`)
5. `streamlit run app.py`

## d. Models used

Metrics are calculated on the 20 percent test split. Because there are 6 classes, Precision, Recall and F1 use weighted average, and AUC is one vs rest weighted.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-|-|-|-|-|-|-|
| Logistic Regression | 0.7551 | 0.9318 | 0.7532 | 0.7551 | 0.7490 | 0.6623 |
| Decision Tree | 0.7710 | 0.9204 | 0.7690 | 0.7710 | 0.7693 | 0.6872 |
| KNN | 0.7648 | 0.9323 | 0.7620 | 0.7648 | 0.7624 | 0.6777 |
| Naive Bayes | 0.6547 | 0.8876 | 0.6621 | 0.6547 | 0.6556 | 0.5359 |
| Support Vector Machine | 0.7734 | 0.9471 | 0.7740 | 0.7734 | 0.7666 | 0.6887 |

### Observations

| ML Model Name | Observation about model performance |
|-|-|
| Logistic Regression | Got 75.5 percent accuracy and an MCC of 0.66. This is a straight line (linear) model, so it cannot capture the sharp jumps between AQI buckets, but it still did decently and its AUC of 0.93 is among the best. It was fast to train and is the easiest to explain. |
| Decision Tree | Accuracy of 77.1 percent and the best F1 (0.7693) of all five models. It handles the bucket boundaries well because it splits on thresholds, which is similar to how the AQI is actually calculated. Its AUC (0.92) is lower than Logistic Regression, KNN and SVM, probably because a tree gives very rough probabilities (the leaves are often pure). With max depth 10 it may overfit a little. |
| KNN | Accuracy 76.5 percent with k = 7 and an AUC of 0.9323, so it ranks classes about as well as Logistic Regression. It depends on scaling, which we did with StandardScaler. Prediction is slower than the other models since it compares each test row with all training rows. |
| Naive Bayes | Clearly the weakest, with accuracy of 65.5 percent and MCC of 0.54. Gaussian Naive Bayes assumes the pollutants are independent and normally distributed. Here PM2.5 and PM10 are strongly correlated (and so are NO, NO2 and NOx), and the values are skewed, so the assumption does not hold. |
| Support Vector Machine | Best accuracy (77.3 percent), best AUC (0.9471), best precision and best MCC (0.6887). The RBF kernel can draw curved boundaries between the buckets. The downside is that it needs more time to train, especially with probability estimates switched on for the AUC score. |

**Overall winner for this dataset: Support Vector Machine.** It is first in accuracy, AUC, precision and MCC. Decision Tree is very close behind (MCC 0.6872 against 0.6887) and even has a slightly higher F1, so the gap is small, but SVM is also better at ranking the classes (AUC), which is why it is picked. Naive Bayes was the worst model on every metric.

**General remarks:** All models stay under about 78 percent accuracy. Part of the reason is that many pollutant readings are missing in the raw data and were filled with the median, so the model does not see the true pollutant that decided the AQI on those days. The neighbouring buckets (for example Poor and Very Poor) also have values that overlap. Checking the wrong predictions on the test set, about 96 to 97 percent of them are only one bucket away from the true one for Logistic Regression, Decision Tree, KNN and SVM (91 percent for Naive Bayes). Since the classes are imbalanced, MCC and weighted F1 are a fairer measure than accuracy alone.
