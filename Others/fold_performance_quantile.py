import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CẤU HÌNH & HÀM HỖ TRỢ
# ==============================================================================
# Tên các file Fold (Giả định cấu trúc tên file chuẩn trong thư mục Cross validation)
# Bạn hãy đảm bảo folder 'Cross validation' nằm cùng cấp với file code này
FOLDS = [1, 2, 3, 4, 5]

def find_folder_containing(filename, search_path='.'):
    """Tìm thư mục chứa file cụ thể"""
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return root
    return None

def calculate_metrics(y_true, y_pred, model_name, fold_num):
    """Tính toán bộ chỉ số Full Metrics cho 1 Fold"""
    # 1. Basic Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mean_sale = np.mean(y_true)
    
    # 2. WAPE
    sum_abs_error = np.sum(np.abs(y_true - y_pred))
    sum_actual = np.sum(y_true)
    wape = sum_abs_error / sum_actual if sum_actual != 0 else 0
    
    # 3. MAPE
    mask = y_true > 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    # 4. Accuracy (Standard)
    accuracy = max(0, 100 - mape)
    
    # 5. Accuracy within 20% (Leader Question)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_diff_pct = np.abs(y_true - y_pred) / y_true
        # Xử lý trường hợp y_true = 0
        abs_diff_pct[y_true == 0] = np.where(y_pred[y_true == 0] == 0, 0, np.inf)
    accuracy_20pct = np.mean(abs_diff_pct <= 0.20) * 100
    
    # 6. Interval Metrics (Normal Approximation)
    std_resid = rmse
    lower_bound = y_pred - 1.96 * std_resid
    upper_bound = y_pred + 1.96 * std_resid
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage_95 = np.mean(in_interval) * 100
    avg_width = np.mean(upper_bound - lower_bound)
    
    return {
        'model': model_name,
        'fold': fold_num,
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
    }

# ==============================================================================
# MAIN PROCESS
# ==============================================================================
print(">>> BẮT ĐẦU QUÁ TRÌNH CROSS-VALIDATION (5 FOLDS)...")

# Tìm đường dẫn file segments
seg_path = 'family_segments.csv'
if not os.path.exists(seg_path):
    # Thử tìm đệ quy
    found_seg = find_folder_containing('family_segments.csv')
    if found_seg: seg_path = os.path.join(found_seg, 'family_segments.csv')
    else: 
        print("❌ LỖI: Không tìm thấy family_segments.csv"); exit()

# Load Segments & Filter Medium
segments = pd.read_csv(seg_path, sep=';')
medium_families = segments[segments['segment'] == 'Medium']['family_name'].unique()
print(f"-> Medium Families: {len(medium_families)}")

# Tìm thư mục chứa Folds
fold_dir = find_folder_containing('train_fold_1.csv')
if not fold_dir:
    print("❌ LỖI: Không tìm thấy thư mục chứa các file fold (train_fold_1.csv...)"); exit()

all_results = []

# VÒNG LẶP QUA 5 FOLD
for fold in FOLDS:
    print(f"\n--- PROCESSING FOLD {fold}/5 ---")
    
    # 1. Load Data Fold
    train_file = os.path.join(fold_dir, f'train_fold_{fold}.csv')
    test_file = os.path.join(fold_dir, f'test_fold_{fold}.csv')
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"⚠️ Cảnh báo: Không tìm thấy file cho fold {fold}, bỏ qua.")
        continue
        
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # 2. Preprocessing (Medium Only)
    train_m = train_df[train_df['family'].isin(medium_families)].copy()
    test_m = test_df[test_df['family'].isin(medium_families)].copy()
    
    train_m = train_m.fillna(0); test_m = test_m.fillna(0)
    train_m['date'] = pd.to_datetime(train_m['date'])
    test_m['date'] = pd.to_datetime(test_m['date'])
    
    # Outlier Clipping
    for fam in medium_families:
        mask = train_m['family'] == fam
        threshold = train_m.loc[mask, 'sales'].quantile(0.99)
        threshold = max(threshold, 10)
        train_m.loc[mask & (train_m['sales'] > threshold), 'sales'] = threshold
        
    # Feature Engineering (Encoding)
    cat_cols = ['family', 'city', 'state', 'type']
    for col in cat_cols:
        le = LabelEncoder()
        all_val = pd.concat([train_m[col], test_m[col]]).unique()
        le.fit(all_val)
        train_m[col] = le.transform(train_m[col])
        test_m[col] = le.transform(test_m[col])
        
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
    
    # ---------------------------------------------------------
    # MODEL 1: XGBOOST
    # ---------------------------------------------------------
    print(f"   > Training XGBoost (Fold {fold})...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror', n_estimators=1000, learning_rate=0.02, # Giảm nhẹ est để chạy nhanh hơn
        max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        early_stopping_rounds=50, eval_metric='mae'
    )
    xgb_model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=0)
    
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_xgb = np.maximum(y_pred_xgb, 0)
    
    res_xgb = calculate_metrics(y_test.values, y_pred_xgb, 'XGBoost', fold)
    all_results.append(res_xgb)
    
    # ---------------------------------------------------------
    # MODEL 2: RANDOM FOREST
    # ---------------------------------------------------------
    print(f"   > Training Random Forest (Fold {fold})...")
    rf_model = RandomForestRegressor(
        n_estimators=200, # Giảm xuống 200 để chạy fold cho nhanh (thực tế 500 tốt hơn)
        max_depth=10, n_jobs=-1, random_state=42, verbose=0
    )
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    y_pred_rf = np.maximum(y_pred_rf, 0)
    
    res_rf = calculate_metrics(y_test.values, y_pred_rf, 'Random Forest', fold)
    all_results.append(res_rf)

# ==============================================================================
# EXPORT
# ==============================================================================
print("\n>>> ĐANG LƯU FILE KẾT QUẢ...")
results_df = pd.DataFrame(all_results)

# Sắp xếp cột
cols_order = ['model', 'fold', 'MAE', 'RMSE', 'MAPE', 'R2', 'accuracy', 'coverage_95', 'avg_width', 'mean_sale', 'WAPE', 'segment', 'accuracy_20pct']
results_df = results_df[cols_order]

print(results_df)
results_df.to_csv('fold_performance_quantile.csv', index=False)

print("\n✅ HOÀN TẤT! File 'fold_performance_quantile.csv' đã được tạo.")