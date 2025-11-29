# os is used for directory and file operations.
import os
# pandas is used to load and manipulate tabular data.
import pandas as pd
# NumPy is used for numerical operations.
import numpy as np
# matplotlib is available for plots (not strictly required here).
import matplotlib.pyplot as plt
# joblib is used to load the encoders.
import joblib
# r2_score is used to compute R-squared per family.
from sklearn.metrics import r2_score

# ---------- Configuration ----------
# Base directory for the project.
BASE_DIR = './'

# Root directory where quantile evaluation outputs are stored.
OUTPUT_DIR = os.path.join(BASE_DIR, 'output/quantile_lstm/')

# Directory where scaler and encoders are stored.
ARTIFACT_DIR = os.path.join(BASE_DIR, 'data/processed_data/artifacts/')

# CSV file that maps family_name -> segment for reporting.
SEGMENT_FILE = os.path.join(BASE_DIR, 'data/processed_data/family_segments.csv')

# Path for per-family performance output.
FAMILY_PERF_PATH = os.path.join(OUTPUT_DIR, 'family_performance_quantile_low.csv')

# Directory containing per-fold predictions from 3_evaluation.py.
PRED_DIR = os.path.join(OUTPUT_DIR, 'fold_predictions')

# ---------- Load encoders ----------
encoders = joblib.load(os.path.join(ARTIFACT_DIR, 'encoders.pkl'))
family_encoder = encoders['family']

# ---------- Load all fold prediction files ----------
pred_files = [
    f for f in os.listdir(PRED_DIR)
    if f.startswith('predictions_fold') and f.endswith('.csv')
]

if not pred_files:
    raise FileNotFoundError(f"No prediction files found in {PRED_DIR}")

all_preds = []
for f in pred_files:
    path = os.path.join(PRED_DIR, f)
    df = pd.read_csv(path)
    all_preds.append(df)

preds_all = pd.concat(all_preds, ignore_index=True)
print(f"Loaded {len(pred_files)} prediction files, total rows = {len(preds_all)}")

# ---------- Compute per-family metrics ----------
def compute_family_metrics(group: pd.DataFrame):
    """
    Compute performance metrics for a single family.
    """
    y_true = group['y_true'].values
    y_q50 = group['q50'].values
    y_q05 = group['q05'].values
    y_q95 = group['q95'].values

    mae = float(np.mean(np.abs(y_true - y_q50)))
    rmse = float(np.sqrt(np.mean((y_true - y_q50) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_q50) / np.maximum(y_true, 1e-6))) * 100.0)

    # R2 can fail if variance of y_true is zero; handle safely.
    if np.var(y_true) > 0:
        r2 = float(r2_score(y_true, y_q50))
    else:
        r2 = np.nan

    inside_20 = np.abs(y_true - y_q50) <= 0.2 * np.maximum(y_true, 1e-6)
    accuracy = float(inside_20.mean() * 100.0)

    # Interval coverage: percentage of true values inside [q05, q95].
    inside = (y_true >= y_q05) & (y_true <= y_q95)
    coverage_95 = float(inside.mean() * 100.0)

    # Average width of the 95% interval.
    avg_width = float(np.mean(y_q95 - y_q05))

    # WAPE = MAE / mean_sale. If mean_sale is zero, WAPE is not defined.
    mean_sale = float(np.mean(y_true))
    if mean_sale > 0:
        wape = float(mae / mean_sale)
    else:
        wape = np.nan

    return pd.Series({
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'accuracy': accuracy,
        'coverage_95': coverage_95,
        'avg_width': avg_width,
        'mean_sale': mean_sale,
        'WAPE': wape,
    })


family_perf = preds_all.groupby('family_name').apply(
    compute_family_metrics
).reset_index()

# Sort by MAE (ascending) to see best-performing families first.
family_perf = family_perf.sort_values('MAE', ascending=True)

# ---------- Attach segment info (High/Medium/Low/Ultra-low) ----------
try:
    seg_df = pd.read_csv(SEGMENT_FILE)
    family_perf = family_perf.merge(
        seg_df[['family_name', 'segment']],
        on='family_name',
        how='left'
    )
except FileNotFoundError:
    print(f"Warning: SEGMENT_FILE not found at {SEGMENT_FILE}. family_perf will not include segment column.")
except Exception as e:
    print(f"Warning: Could not merge segment info: {e}")

# Save per-family performance metrics to CSV.
family_perf.to_csv(FAMILY_PERF_PATH, index=False)
print(f"Saved family performance metrics to: {FAMILY_PERF_PATH}")

