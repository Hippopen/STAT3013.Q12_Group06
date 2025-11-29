import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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
print(">>> [2/5] Feature Engineering & Outlier Clipping...")
for fam in medium_families:
    mask = train_m['family'] == fam
    threshold = train_m.loc[mask, 'sales'].quantile(0.99)
    threshold = max(threshold, 10) 
    train_m.loc[mask & (train_m['sales'] > threshold), 'sales'] = threshold

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
weights = train_m['year'].apply(lambda x: 1.5 if x == 2015 else 1.0)

# ==============================================================================
# 2. TRAINING XGBOOST
# ==============================================================================
print(">>> [3/5] Training XGBoost (Median Regression)...")

model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=100,
    eval_metric='mae'
)

model.fit(
    X_train, y_train,
    sample_weight=weights,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=0
)

# ==============================================================================
# 3. EVALUATION (LEADER'S REQUESTED METRICS)
# ==============================================================================
print(">>> [4/5] Calculating Full Metrics...")

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
    
    # Basic Metrics
    mae = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    r2 = r2_score(y_true_f, y_pred_f)
    mean_sale = np.mean(y_true_f)
    
    # WAPE
    sum_abs_error = np.sum(np.abs(y_true_f - y_pred_f))
    sum_actual = np.sum(y_true_f)
    wape = sum_abs_error / sum_actual if sum_actual != 0 else 0
    
    # MAPE
    mask = y_true_f > 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true_f[mask] - y_pred_f[mask]) / y_true_f[mask])) * 100
    else:
        mape = 0.0
    
    # Accuracy (Theo file mẫu: 100 - MAPE)
    accuracy = max(0, 100 - mape)
    
    # --- CÂU TRẢ LỜI CHO LEADER: Accuracy within 20% ---
    # Tỷ lệ số ngày mà sai số tuyệt đối <= 20% giá trị thực tế
    # Với y_true = 0: nếu y_pred = 0 thì đúng, ngược lại là sai
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_diff_pct = np.abs(y_true_f - y_pred_f) / y_true_f
        # Xử lý trường hợp y_true = 0
        # Nếu y_true=0 và y_pred=0 -> diff=0 (Pass). Nếu y_pred>0 -> diff=inf (Fail)
        abs_diff_pct[y_true_f == 0] = np.where(y_pred_f[y_true_f == 0] == 0, 0, np.inf)
        
    accuracy_20pct = np.mean(abs_diff_pct <= 0.20) * 100
    
    # Interval Metrics (Normal Approximation for Point Forecast)
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
        'accuracy': accuracy,          # Format file mẫu (100 - MAPE)
        'coverage_95': coverage_95,
        'avg_width': avg_width,
        'mean_sale': mean_sale,
        'WAPE': wape,
        'segment': 'Medium',
        'accuracy_20pct': accuracy_20pct # <--- Metric Leader hỏi thêm
    })

results_df = pd.DataFrame(results)
# Sắp xếp cột chuẩn file mẫu + cột mới
cols = ['family_name', 'MAE', 'RMSE', 'MAPE', 'R2', 'accuracy', 'coverage_95', 'avg_width', 'mean_sale', 'WAPE', 'segment', 'accuracy_20pct']
results_df = results_df[cols]

print("\n=== KẾT QUẢ XGBOOST (FULL + LEADER QUESTION) ===")
print(results_df.head())

# ==============================================================================
# 4. EXPORT
# ==============================================================================
print(">>> [5/5] Saving files...")

results_df.to_csv('xgb_family_performance_M.csv', index=False)

importance = model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feat_df = feat_df.sort_values(by='Importance', ascending=False)
feat_df.to_csv('xgb_feature_importance.csv', index=False)

# File so sánh tổng hợp
comparison_file = 'm_group_model_comparison.csv'
final_comp = results_df.copy()
final_comp['model'] = 'XGBoost'
final_comp = final_comp[['family_name', 'model'] + [c for c in cols if c != 'family_name']]
final_comp.to_csv(comparison_file, index=False)

print("\n✅ HOÀN TẤT! Đã thêm cột 'accuracy_20pct' trả lời câu hỏi của Leader.")