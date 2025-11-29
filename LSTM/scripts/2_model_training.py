# os is used for file path and directory operations.
import os
# NumPy is used for numerical operations and array handling.
import numpy as np
# pandas is used to load and manipulate tabular data.
import pandas as pd
# TensorFlow and Keras are used to build and train the neural network.
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
# joblib is used to load and save Python objects (scaler, encoders, target scale).
import joblib

# ---------- Configuration ----------
# FOLD_ID selects which time-based fold to train on (1 to 5).
FOLD_ID = 5

# Sequence length (number of past time steps) used as input to the LSTM.
SEQ_LEN = 30

# Maximum number of training epochs.
EPOCHS = 60

# Batch size used during training.
BATCH_SIZE = 256

# Base directory for the project.
BASE_DIR = './'

# Directory containing the processed time-based folds.
FOLD_DIR = os.path.join(BASE_DIR, 'data/processed_data/folds/')

# Directory where trained models will be saved.
MODEL_DIR = os.path.join(BASE_DIR, 'models/quantile_lstm/')

# Directory containing scaler and encoders saved during data processing.
ARTIFACT_DIR = os.path.join(BASE_DIR, 'data/processed_data/artifacts/')

# ---------- Segment configuration ----------
# SEGMENT controls which family segment to train on.
#   None       -> train on ALL families (default, same behavior as before)
#   'High'     -> only families with segment == 'High'
#   'Medium'   -> only families with segment == 'Medium'
#   'Low'      -> only families with segment == 'Low'
#   'Ultra-low'-> only families with segment == 'Ultra-low'
SEGMENT = 'Low'

# CSV file that maps family_name -> segment (High/Medium/Low/Ultra-low).
# You can regenerate this file from train_data using your preprocessing step.
SEGMENT_FILE = os.path.join(BASE_DIR, 'data/processed_data/family_segments.csv')

# Create the model directory if it does not exist.
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Feature configuration ----------
# These are the numeric columns used as inputs to the LSTM.
NUMERIC_COLS = [
    'onpromotion', 'dcoilwtico',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    'rolling_mean_7', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_30',
]


def quantile_loss(q):
    """
    Create a quantile loss function for quantile q.

    For a given quantile q in (0, 1), the loss is:
        L_q(y, y_hat) = mean( max(q * (y - y_hat), (q - 1) * (y - y_hat)) )

    This penalizes underestimation vs overestimation asymmetrically.
    """
    def loss(y_true, y_pred):
        # e = y_true - y_pred is the error.
        e = y_true - y_pred
        # K.maximum applies the quantile loss element-wise.
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
    return loss


def create_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Create input sequences and targets for the LSTM model.

    For each (store_nbr, family) pair:
      1. Sort by date.
      2. Extract NUMERIC_COLS as inputs.
      3. Build sliding windows of length seq_len.
      4. Target y_real is the current row's 'sales' (real units).

    Returns:
        X: NumPy array of shape (num_samples, seq_len, num_features)
        y_real: NumPy array of shape (num_samples,) in real sales units
    """
    all_X = []
    all_y_real = []

    # Group by store and family to respect panel structure.
    for (store_id, family_id), group in df.groupby(['store_nbr', 'family']):
        # Sort each series chronologically.
        group = group.sort_values('date')

        # Extract numeric feature values and target sales.
        values = group[NUMERIC_COLS].values
        targets_real = group['sales'].values

        # Skip groups that are too short.
        if len(group) <= seq_len:
            continue

        # Build rolling windows.
        for i in range(seq_len, len(group)):
            all_X.append(values[i - seq_len:i])
            all_y_real.append(targets_real[i])

    # Convert lists to NumPy arrays.
    X = np.array(all_X)
    y_real = np.array(all_y_real)

    return X, y_real


# ---------- Load artifacts ----------
# Load encoders (used to get 'family' encoder for segment filtering).
encoders = joblib.load(os.path.join(ARTIFACT_DIR, 'encoders.pkl'))

# ---------- Load fold data ----------
# Paths to training and validation data for this fold.
train_path = os.path.join(FOLD_DIR, f'train_fold_{FOLD_ID}.csv')
val_path = os.path.join(FOLD_DIR, f'test_fold_{FOLD_ID}.csv')

# Load the data with 'date' parsed as datetime.
train_df = pd.read_csv(train_path, parse_dates=['date'])
val_df = pd.read_csv(val_path, parse_dates=['date'])

family_encoder = encoders.get('family')

if SEGMENT is not None:
    # Load family -> segment mapping.
    seg_df = pd.read_csv(SEGMENT_FILE)

    # Get all family names that belong to the requested segment.
    fam_names = (
        seg_df.loc[seg_df['segment'] == SEGMENT, 'family_name']
        .astype(str)
        .unique()
    )

    if len(fam_names) == 0:
        raise ValueError(f"Không tìm thấy family nào có segment = {SEGMENT} trong file {SEGMENT_FILE}")

    if family_encoder is None:
        raise ValueError("Encoder for 'family' không tồn tại trong encoders.pkl")

    # Map family_name -> encoded family id used in processed folds.
    fam_ids = family_encoder.transform(fam_names)

    # Filter train/val theo các family thuộc segment đó.
    train_df = train_df[train_df['family'].isin(fam_ids)].copy()
    val_df = val_df[val_df['family'].isin(fam_ids)].copy()

    print(f"Training on SEGMENT = {SEGMENT}, fold {FOLD_ID}")
    print(f"Số family trong segment: {len(fam_names)}")
    print(f"Train rows after segment filter: {len(train_df)}, validation rows: {len(val_df)}")
else:
    print(f"Training on ALL families, fold {FOLD_ID}")
    print(f"Train rows: {len(train_df)}, validation rows: {len(val_df)}")


# ---------- Create sequences ----------
X_train, y_train_real = create_sequences(train_df, seq_len=SEQ_LEN)
print(f"X_train shape: {X_train.shape}, y_train_real shape: {y_train_real.shape}")

X_val, y_val_real = create_sequences(val_df, seq_len=SEQ_LEN)
print(f"X_val shape: {X_val.shape}, y_val_real shape: {y_val_real.shape}")

# ---------- Scale the target ----------
# We keep X as-is (features already scaled in Phase 1). Here we only scale the target.
# For stability we scale by the 95th percentile of training sales, with a minimum of 1.0.
target_scale = float(max(np.quantile(y_train_real, 0.95), 1.0))
print(f"Target scale for fold {FOLD_ID}: {target_scale:.4f}")

# Save the target scale so that evaluation can use the same value.
scale_path = os.path.join(MODEL_DIR, f'target_scale_fold{FOLD_ID}.pkl')
joblib.dump(target_scale, scale_path)
print(f"Saved target scale to: {scale_path}")

# Scale the targets.
y_train_scaled = y_train_real / target_scale
y_val_scaled = y_val_real / target_scale

# Ensure non-negative scaled values (sales should be >= 0).
y_train_scaled = np.maximum(y_train_scaled, 0.0)
y_val_scaled = np.maximum(y_val_scaled, 0.0)

# ---------- Build the Quantile LSTM model ----------
# Input layer: expects sequences of length SEQ_LEN with len(NUMERIC_COLS) features.
input_layer = Input(shape=(SEQ_LEN, len(NUMERIC_COLS)), name='input_sequence')

# Shared LSTM backbone.
x = LSTM(128, return_sequences=True)(input_layer)
x = Dropout(0.3)(x)
x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.2)(x)

# Three quantile outputs: 5%, 50%, 95%.
# ReLU activation enforces non-negative predictions on the scaled space.
q05 = Dense(1, activation='relu', name='q05')(x)
q50 = Dense(1, activation='relu', name='q50')(x)
q95 = Dense(1, activation='relu', name='q95')(x)

# Build the model with three outputs.
model = Model(inputs=input_layer, outputs=[q05, q50, q95])

# Compile the model with quantile losses.
model.compile(
    optimizer='adam',
    loss={
        'q05': quantile_loss(0.05),
        'q50': quantile_loss(0.5),
        'q95': quantile_loss(0.95),
    },
    loss_weights={
        'q05': 1.0,
        'q50': 1.0,
        'q95': 1.0,
    },
)

# Print model summary for inspection.
model.summary()

# ---------- Training ----------
# Define path where the best model for this fold will be saved.
model_path = os.path.join(MODEL_DIR, f'quantile_lstm_fold{FOLD_ID}.keras')

# Early stopping to stop training when validation loss stops improving.
early_stopping = EarlyStopping(
    patience=10,
    restore_best_weights=True,
)

# Model checkpoint to save the best model based on validation loss.
model_checkpoint = ModelCheckpoint(
    model_path,
    save_best_only=True,
)

callbacks = [early_stopping, model_checkpoint]

# Training with the same target y for all three quantile outputs.
y_train_dict = {'q05': y_train_scaled, 'q50': y_train_scaled, 'q95': y_train_scaled}
y_val_dict = {'q05': y_val_scaled, 'q50': y_val_scaled, 'q95': y_val_scaled}

model.fit(
    X_train,
    y_train_dict,
    validation_data=(X_val, y_val_dict),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

print(f"Training completed. Best model saved to: {model_path}")
