# explore.py
# First look at city_day.csv before building any model.
# Checks size, missing values, class balance and how the pollutants relate to each other.
#
# How to run:  python explore.py

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "city_day.csv"
if len(sys.argv) > 1:
    DATA_FILE = sys.argv[1]

PLOT_DIR = "eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
              "O3", "Benzene", "Toluene", "Xylene"]

df = pd.read_csv(DATA_FILE)

# 1. basic info
print("Shape:", df.shape)
print("\nColumns and data types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])
print("\nDate range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("Number of cities:", df["City"].nunique())
print("\nRows per city:")
print(df["City"].value_counts())

# 2. missing values
# a lot of the pollutant columns have gaps, this decides how we handle them later
missing_pct = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)
print("\nMissing values (percent of rows):")
print(missing_pct)

plt.figure(figsize=(8, 5))
sns.barplot(x=missing_pct.values, y=missing_pct.index, color="steelblue")
plt.xlabel("Percent missing")
plt.title("Missing values in each column")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "missing_values.png"), dpi=120)
plt.close()

# 3. target column
print("\nAQI_Bucket counts:")
print(df["AQI_Bucket"].value_counts(dropna=False))

order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="AQI_Bucket", order=order, color="darkorange")
plt.title("Class balance of AQI_Bucket")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "class_balance.png"), dpi=120)
plt.close()

# rows where the bucket is missing cannot be used for training
print("\nRows with no AQI_Bucket:", df["AQI_Bucket"].isnull().sum())

# 4. does AQI decide the bucket? (this is why AQI cannot be used as a feature)
print("\nAQI range inside each bucket:")
print(df.groupby("AQI_Bucket")["AQI"].agg(["min", "max", "count"]).reindex(order))

# 5. summary statistics of the pollutants
print("\nPollutant summary:")
print(df[pollutants].describe().round(2))

# 6. correlation between pollutants
plt.figure(figsize=(9, 7))
sns.heatmap(df[pollutants].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation between pollutants")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlation.png"), dpi=120)
plt.close()

# 7. average PM2.5 by month, to check if winter is really worse
df["Month"] = df["Date"].dt.month
monthly = df.groupby("Month")["PM2.5"].mean()
print("\nAverage PM2.5 by month:")
print(monthly.round(1))

plt.figure(figsize=(7, 4))
monthly.plot(kind="bar", color="seagreen")
plt.ylabel("Average PM2.5")
plt.title("PM2.5 by month")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pm25_by_month.png"), dpi=120)
plt.close()

# 8. which cities have the most Severe days?
# useful to know if one city dominates a class
severe_by_city = (df[df["AQI_Bucket"] == "Severe"]["City"].value_counts().head(8))
print("\nCities with the most Severe days:")
print(severe_by_city)

print("\nPlots saved in the", PLOT_DIR, "folder")
