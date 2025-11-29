import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================
def find_file_path(filename, search_path='.'):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print(">>> [1/5] Loading Data...")

SEGMENT_PATH = find_file_path('family_segments.csv')
TRAIN_FOLD_PATH = find_file_path('train_fold_1.csv')
TEST_FOLD_PATH = find_file_path('test_fold_1.csv')

if not SEGMENT_PATH or not TRAIN_FOLD_PATH or not TEST_FOLD_PATH:
    print("❌ LỖI: Không tìm thấy file dữ liệu.")
    exit()

segments = pd.read_csv(SEGMENT_PATH, sep=';')
medium_families = segments[segments['segment'] == 'Medium']['family_name'].unique()

train_df = pd.read_csv(TRAIN_FOLD_PATH)
test_df = pd.read_csv(TEST_FOLD_PATH)

train_m = train_df[train_df['family'].isin(medium_families)].copy()
test_m = test_df[test_df['family'].isin(medium_families)].copy()

train_m = train_m.fillna(0)
test_m = test_m.fillna(0)
train_m['date'] = pd.to_datetime(train_m['date'])
test_m['date'] = pd.to_datetime(test_m['date'])

# Clipping Outliers
for fam in medium_families:
    mask = train_m['family'] == fam
    threshold = train_m.loc[mask, 'sales'].quantile(0.99)
    threshold = max(threshold, 10) 
    train_m.loc[mask & (train_m['sales'] > threshold), 'sales'] = threshold

# Feature Engineering
cat_cols = ['family', 'city', 'state', 'type']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train_m[col], test_m[col]]).unique()
    le.fit(all_values)
    train_m[col] = le.transform(train_m[col])
    test_m[col] = le.transform(test_m[col])
    encoders[col] = le

features = [
    'store_nbr', 'family', 'onpromotion', 'transactions', 
    'city', 'state', 'type', 'cluster', 'dcoilwtico', 
    'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend',
    'sales_lag_7', 'sales_lag_14', 'rolling_mean_30'
]

X_train = train_m[features]
y_train = train_m['sales']
X_test = test_m[features]
y_test = test_m['sales']

# ==============================================================================
# 2. TRAINING RANDOM FOREST
# ==============================================================================
print(">>> [2/5] Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

# ==============================================================================
# 3. EVALUATION
# ==============================================================================
print(">>> [3/5] Calculating Full Metrics...")

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

test_m['predicted_sales'] = y_pred
test_m['family_name'] = encoders['family'].inverse_transform(test_m['family'])

results = []
unique_families = test_m['family_name'].unique()

for fam in unique_families:
    df_fam = test_m[test_m['family_name'] == fam]
    y_true_f = df_fam['sales'].values
    y_pred_f = df_fam['predicted_sales'].values
    
    mae = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    r2 = r2_score(y_true_f, y_pred_f)
    mean_sale = np.mean(y_true_f)
    
    sum_abs_error = np.sum(np.abs(y_true_f - y_pred_f))
    sum_actual = np.sum(y_true_f)
    wape = sum_abs_error / sum_actual if sum_actual != 0 else 0
    
    mask = y_true_f > 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true_f[mask] - y_pred_f[mask]) / y_true_f[mask])) * 100
    else:
        mape = 0.0
    
    # Accuracy (File mẫu: 100 - MAPE)
    accuracy = max(0, 100 - mape)
    
    # Accuracy within 20% (Leader Question)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_diff_pct = np.abs(y_true_f - y_pred_f) / y_true_f
        abs_diff_pct[y_true_f == 0] = np.where(y_pred_f[y_true_f == 0] == 0, 0, np.inf)
    accuracy_20pct = np.mean(abs_diff_pct <= 0.20) * 100
    
    # Interval Metrics
    std_resid = rmse
    lower_bound = y_pred_f - 1.96 * std_resid
    upper_bound = y_pred_f + 1.96 * std_resid
    in_interval = (y_true_f >= lower_bound) & (y_true_f <= upper_bound)
    coverage_95 = np.mean(in_interval) * 100
    avg_width = np.mean(upper_bound - lower_bound)
    
    results.append({
        'family_name': fam,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'accuracy': accuracy,
        'coverage_95': coverage_95,
        'avg_width': avg_width,
        'mean_sale': mean_sale,
        'WAPE': wape,
        'segment': 'Medium',
        'accuracy_20pct': accuracy_20pct
    })

results_df = pd.DataFrame(results)
cols = ['family_name', 'MAE', 'RMSE', 'MAPE', 'R2', 'accuracy', 'coverage_95', 'avg_width', 'mean_sale', 'WAPE', 'segment', 'accuracy_20pct']
results_df = results_df[cols]

print("\n=== KẾT QUẢ RANDOM FOREST (FULL + LEADER QUESTION) ===")
print(results_df.head())

# ==============================================================================
# 4. EXPORT & MERGE
# ==============================================================================
print(">>> [4/5] Saving files...")

results_df.to_csv('rf_family_performance_M.csv', index=False)

comparison_file = 'm_group_model_comparison.csv'
new_data = results_df.copy()
new_data['model'] = 'Random Forest'
new_data = new_data[['family_name', 'model'] + [c for c in cols if c != 'family_name']]

if os.path.exists(comparison_file):
    print(f"-> Đang nối vào file '{comparison_file}'...")
    old_df = pd.read_csv(comparison_file)
    old_df = old_df[old_df['model'] != 'Random Forest']
    final_df = pd.concat([old_df, new_data], ignore_index=True)
else:
    final_df = new_data

final_df.to_csv(comparison_file, index=False)

print("\n✅ HOÀN TẤT! Đã update cả 2 model.")