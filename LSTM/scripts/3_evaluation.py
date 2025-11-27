import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------- Configuration ----------
FOLD_ID = 5

BASE_DIR = './'
FOLD_DIR = os.path.join(BASE_DIR, 'data/processed_data/folds/')
MODEL_DIR = os.path.join(BASE_DIR, 'models/quantile_lstm/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output/quantile_lstm/')
ARTIFACT_DIR = os.path.join(BASE_DIR, 'data/processed_data/artifacts/')

# ---------- Segment configuration ----------
# SEGMENT controls which family segment to evaluate on.
#   None       -> evaluate on ALL families (default, same behavior as trước đây)
#   'High'     -> chỉ evaluate các family thuộc segment High
#   'Medium'   -> chỉ evaluate các family thuộc segment Medium
#   'Low'      -> chỉ evaluate các family thuộc segment Low
#   'Ultra-low'-> chỉ evaluate các family thuộc segment Ultra-low
SEGMENT = 'Low'

# CSV file that maps family_name -> segment (High/Medium/Low/Ultra-low).
SEGMENT_FILE = os.path.join(BASE_DIR, 'data/processed_data/family_segments.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
PRED_DIR = os.path.join(OUTPUT_DIR, 'fold_predictions')
os.makedirs(PRED_DIR, exist_ok=True)

# Load encoders (to decode family IDs và filter theo segment).
encoders = joblib.load(os.path.join(ARTIFACT_DIR, 'encoders.pkl'))

# ---------- Quantile loss (không dùng trực tiếp nhưng để reference) ----------
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return np.mean(np.maximum(q * e, (q - 1) * e), axis=-1)
    return loss

# ---------- Load model and target scale ----------
model_path = os.path.join(MODEL_DIR, f'quantile_lstm_fold{FOLD_ID}.keras')
model = load_model(model_path, compile=False)
print(f"Loaded model from: {model_path}")

scale_path = os.path.join(MODEL_DIR, f'target_scale_fold{FOLD_ID}.pkl')
target_scale = float(joblib.load(scale_path))
print(f"Loaded target scale from: {scale_path} (value = {target_scale:.4f})")

# ---------- Helper: create sequences for evaluation ----------
SEQ_LEN = 30  # phải trùng với training
NUMERIC_COLS = [
    'onpromotion', 'dcoilwtico',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    'rolling_mean_7', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_30',
]


def create_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Giống training: build sequences X và giữ y_true + family_id.
    """
    all_X = []
    all_y = []
    all_family_ids = []

    for (store_id, family_id), group in df.groupby(['store_nbr', 'family']):
        group = group.sort_values('date')

        values = group[NUMERIC_COLS].values
        targets = group['sales'].values

        if len(group) <= seq_len:
            continue

        for i in range(seq_len, len(group)):
            all_X.append(values[i - seq_len:i])
            all_y.append(targets[i])
            all_family_ids.append(family_id)

    if len(all_X) == 0:
        return np.empty((0, seq_len, len(NUMERIC_COLS))), np.array([]), np.array([])

    X = np.array(all_X)
    y = np.array(all_y)
    family_ids = np.array(all_family_ids)

    return X, y, family_ids


# ---------- Load test data ----------
test_path = os.path.join(FOLD_DIR, f'test_fold_{FOLD_ID}.csv')
test_df = pd.read_csv(test_path, parse_dates=['date'])

family_encoder = encoders.get('family')

if SEGMENT is not None:
    # Load family -> segment mapping.
    seg_df = pd.read_csv(SEGMENT_FILE)

    fam_names = (
        seg_df.loc[seg_df['segment'] == SEGMENT, 'family_name']
        .astype(str)
        .unique()
    )

    if len(fam_names) == 0:
        raise ValueError(f"Không tìm thấy family nào có segment = {SEGMENT} trong file {SEGMENT_FILE}")

    if family_encoder is None:
        raise ValueError("Encoder for 'family' không tồn tại trong encoders.pkl")

    fam_ids = family_encoder.transform(fam_names)

    before_rows = len(test_df)
    test_df = test_df[test_df['family'].isin(fam_ids)].copy()

    print(f"Evaluating Quantile LSTM on SEGMENT = {SEGMENT}, fold {FOLD_ID}")
    print(f"Test rows: {before_rows} -> {len(test_df)} sau khi filter theo segment")
else:
    print(f"Evaluating Quantile LSTM on ALL families, fold {FOLD_ID}")
    print(f"Test rows: {len(test_df)}")

# ---------- Build sequences ----------
X_test, y_true, family_ids = create_sequences(test_df, seq_len=SEQ_LEN)
print(f"X_test shape: {X_test.shape}, y_true shape: {y_true.shape}")

if X_test.shape[0] == 0:
    print("Không có sample nào sau khi tạo sequence (có thể do filter segment quá hẹp hoặc data quá ngắn).")
    # Lưu file metrics rỗng để script khác không bị crash
    metrics_df = pd.DataFrame([{
        'fold': FOLD_ID,
        'segment': SEGMENT if SEGMENT is not None else 'ALL',
        'MAE': np.nan,
        'RMSE': np.nan,
        'MAPE': np.nan,
        'R2': np.nan,
        'accuracy': np.nan,
        'coverage_95': np.nan,
        'avg_width': np.nan,
    }])
    metrics_path = os.path.join(OUTPUT_DIR, f'eval_metrics_fold{FOLD_ID}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved empty metrics to: {metrics_path}")
    raise SystemExit(0)

# ---------- Predict ----------
q05_pred, q50_pred, q95_pred = model.predict(X_test, verbose=1)

# Rescale back to real units.
y_true_real = y_true
q05_real = q05_pred.flatten() * target_scale
q50_real = q50_pred.flatten() * target_scale
q95_real = q95_pred.flatten() * target_scale

# ---------- Compute global metrics ----------
mae = mean_absolute_error(y_true_real, q50_real)

# FIX: scikit-learn version cũ không hỗ trợ squared=False
mse = mean_squared_error(y_true_real, q50_real)
rmse = float(np.sqrt(mse))

mape = float(
    np.mean(
        np.abs((y_true_real - q50_real) / np.maximum(y_true_real, 1e-6))
    ) * 100.0
)

try:
    r2 = float(r2_score(y_true_real, q50_real))
except Exception:
    r2 = np.nan

# Accuracy: percentage of predictions within 20% of true value.
within_20 = np.abs(y_true_real - q50_real) <= 0.2 * np.maximum(y_true_real, 1e-6)
accuracy = float(np.mean(within_20) * 100.0)

# Coverage: percentage of true values inside [q05, q95].
inside_95 = (y_true_real >= q05_real) & (y_true_real <= q95_real)
coverage_95 = float(np.mean(inside_95) * 100.0)

# Average width of the 95% interval.
avg_width = float(np.mean(q95_real - q05_real))

print("Global metrics:")
print(f"MAE         : {mae:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"MAPE        : {mape:.2f}%")
print(f"R2          : {r2:.4f}")
print(f"Accuracy    : {accuracy:.2f}%")
print(f"Coverage 95 : {coverage_95:.2f}%")
print(f"Avg width   : {avg_width:.4f}")

metrics_df = pd.DataFrame([{
    'fold': FOLD_ID,
    'segment': SEGMENT if SEGMENT is not None else 'ALL',
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'R2': r2,
    'accuracy': accuracy,
    'coverage_95': coverage_95,
    'avg_width': avg_width,
}])

metrics_path = os.path.join(OUTPUT_DIR, f'eval_metrics_fold{FOLD_ID}.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved fold-level metrics to: {metrics_path}")

# ---------- Save detailed predictions per row (for summary script) ----------
# Decode family IDs back to names for easier analysis.
family_encoder = encoders['family']
family_names = family_encoder.inverse_transform(family_ids)

preds_df = pd.DataFrame({
    'fold': FOLD_ID,
    'family_id': family_ids,
    'family_name': family_names,
    'y_true': y_true_real,
    'q05': q05_real,
    'q50': q50_real,
    'q95': q95_real,
})

pred_path = os.path.join(PRED_DIR, f'predictions_fold{FOLD_ID}.csv')
preds_df.to_csv(pred_path, index=False)
print(f"Saved detailed predictions to: {pred_path}")
