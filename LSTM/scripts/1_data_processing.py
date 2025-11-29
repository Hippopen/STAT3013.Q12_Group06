# The os module is used to work with file paths and directories.
import os
# pandas is used for loading and manipulating tabular data.
import pandas as pd
# NumPy is used for numerical operations.
import numpy as np
# MinMaxScaler scales numeric features into a fixed range [0, 1].
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# joblib is used to save and load Python objects (scaler, encoders).
import joblib

# ---------- Configuration ----------
# RAW_PATH: path to the original training data (before any processing).
RAW_PATH = './data/raw_data/train_data.csv'

# FOLD_DIR: directory where time-based folds will be stored as CSV files.
FOLD_DIR = './data/processed_data/folds/'

# ARTIFACT_DIR: directory where the scaler and encoders will be stored.
ARTIFACT_DIR = './data/processed_data/artifacts/'

# Create directories if they do not exist.
os.makedirs(FOLD_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------- Load raw data ----------
# Read the raw dataset. The 'date' column is parsed as datetime.
df = pd.read_csv(RAW_PATH, parse_dates=['date'])

# Sort by store, family, and date to ensure proper time ordering.
df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

# ---------- Feature Engineering ----------
# Create lag features for sales. These are shifted versions of the sales series.
df['sales_lag_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
df['sales_lag_14'] = df.groupby(['store_nbr', 'family'])['sales'].shift(14)
df['sales_lag_30'] = df.groupby(['store_nbr', 'family'])['sales'].shift(30)

# Create rolling statistics based on past sales (using a one-step shift to avoid leakage).
df['rolling_mean_7'] = (
    df.groupby(['store_nbr', 'family'])['sales']
      .shift(1)
      .rolling(7)
      .mean()
)
df['rolling_mean_30'] = (
    df.groupby(['store_nbr', 'family'])['sales']
      .shift(1)
      .rolling(30)
      .mean()
)
df['rolling_std_7'] = (
    df.groupby(['store_nbr', 'family'])['sales']
      .shift(1)
      .rolling(7)
      .std()
)
df['rolling_std_30'] = (
    df.groupby(['store_nbr', 'family'])['sales']
      .shift(1)
      .rolling(30)
      .std()
)

# Remove any rows that contain NaN values created by lag and rolling operations.
df = df.dropna().reset_index(drop=True)

# At this point, 'sales' remains in real units (no log transform).

# ---------- Scale numeric features ----------
# Define the numeric columns that will be scaled using MinMaxScaler.
NUMERIC_COLS = [
    'onpromotion', 'dcoilwtico',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    'rolling_mean_7', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_30',
]

# Fit a MinMaxScaler on the numeric features and transform them.
scaler = MinMaxScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

# ---------- Encode categorical features ----------
# These categorical columns will be label-encoded and their encoders stored.
CAT_COLS = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
encoders = {}

for col in CAT_COLS:
    # Create a LabelEncoder for this column.
    le = LabelEncoder()
    # Fit the encoder on the string representation of the column.
    df[col] = le.fit_transform(df[col].astype(str))
    # Store the encoder so that the mapping can be reused later.
    encoders[col] = le

# ---------- Save artifacts ----------
# Save the scaler and encoders to disk for later use during training and evaluation.
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler.pkl'))
joblib.dump(encoders, os.path.join(ARTIFACT_DIR, 'encoders.pkl'))
print("Saved scaler and encoders to artifact directory.")

# ---------- Create time-based folds ----------
# Each fold is defined by a training period and a validation (test) period.
folds = [
    (1, '2013-01-01', '2015-12-31', '2016-01-01', '2016-03-31'),
    (2, '2013-01-01', '2016-03-31', '2016-04-01', '2016-06-30'),
    (3, '2013-01-01', '2016-06-30', '2016-07-01', '2016-09-30'),
    (4, '2013-01-01', '2016-09-30', '2016-10-01', '2016-12-31'),
    (5, '2013-01-01', '2016-12-31', '2017-01-01', '2017-03-31'),
]

# Generate and save train/test CSV files for each fold.
for fold_id, tr_start, tr_end, te_start, te_end in folds:
    train_mask = (df['date'] >= tr_start) & (df['date'] <= tr_end)
    test_mask = (df['date'] >= te_start) & (df['date'] <= te_end)

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    train_path = os.path.join(FOLD_DIR, f'train_fold_{fold_id}.csv')
    test_path = os.path.join(FOLD_DIR, f'test_fold_{fold_id}.csv')

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Saved fold {fold_id}: train rows = {len(df_train)}, test rows = {len(df_test)}")

print("Data processing completed.")
