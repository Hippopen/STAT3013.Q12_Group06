import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import time

#  1. Cài đặt Cấu hình Chung 

HIGH_VOLUME_FAMILIES = [
    'GROCERY I', 'PRODUCE', 'BEVERAGES', 'DAIRY',
    'BREAD/BAKERY', 'DELI', 'EGGS'
]

N_FOLDS = 5
target = 'sales'

features_high = [
    'store_nbr', 'family', 'onpromotion', 'transactions', 
    'city', 'state', 'type', 'cluster', 'dcoilwtico', 
    'is_holiday', 'day_of_week', 'week_of_year', 'month', 'year', 
    'is_weekend', 'sales_lag_7', 'sales_lag_14', 'rolling_mean_30'
]
categorical_features_high = [
    'store_nbr', 'family', 'city', 'state', 'type', 
    'cluster', 'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend'
]

features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster', 
    'dcoilwtico', 'is_holiday', 'day_of_week', 'week_of_year', 'month', 'year', 
    'is_weekend'
]
categorical_features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 
    'cluster', 'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend'
]

all_results_dfs = []

print("Bắt đầu PIPELINE 'CHIA ĐỂ TRỊ' TỔNG HỢP (Nâng cấp: có Sample Weights)...")
start_time = time.time()

#  2. Chạy Vòng lặp 5-Fold Cross-Validation 

for fold in range(1, N_FOLDS + 1):
    print(f" Đang chạy Fold {fold}/{N_FOLDS} ")
    
    # 1. Load data
    train_file = f'folds/train_fold_{fold}.csv'
    test_file = f'folds/test_fold_{fold}.csv'
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Tính Sample Weights
    print(f"Fold {fold}: Đang tính toán Sample Weights (Trọng số)...")
    family_sales_sum = df_train.groupby('family')['sales'].sum()
    family_weights_map = family_sales_sum / family_sales_sum.mean()
    
    # 2. Tách data
    print(f"Fold {fold}: Đang tách data (High/Low Volume)...")
    df_train_high = df_train[df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test_high = df_test[df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    
    df_train_low = df_train[~df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test_low = df_test[~df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()

    #  3. Xử lý và Train MÔ HÌNH 1 (High-Volume) 
    
    print(f"Fold {fold}: Đang xử lý Mô hình 1 (High-Volume)...")
    df_train_high[features_high] = df_train_high[features_high].fillna(0)
    df_test_high[features_high] = df_test_high[features_high].fillna(0)
    
    for col in categorical_features_high:
        all_categories = pd.concat([df_train_high[col], df_test_high[col]]).unique()
        df_train_high[col] = pd.Categorical(df_train_high[col], categories=all_categories)
        df_test_high[col] = pd.Categorical(df_test_high[col], categories=all_categories)

    df_train_high['sample_weight'] = df_train_high['family'].map(family_weights_map)
    y_train_weights = df_train_high['sample_weight']

    X_train_high = df_train_high[features_high]
    X_test_high = df_test_high[features_high]
    
    y_train_high_log = np.log1p(df_train_high[target])
    y_test_high_orig = df_test_high[target]
    
    xgb_model_1 = xgb.XGBRegressor(
        objective='reg:squarederror', eval_metric='rmse', n_estimators=2000,
        learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, early_stopping_rounds=100, enable_categorical=True
    )
    
    print(f"Fold {fold}: Đang train Mô hình 1 (Log + Weights)...")
    xgb_model_1.fit(
        X_train_high, y_train_high_log,
        sample_weight=y_train_weights,
        eval_set=[(X_test_high, np.log1p(y_test_high_orig))],
        verbose=False
    )
    
    preds_high_log = xgb_model_1.predict(X_test_high)
    preds_high_orig = np.maximum(0, np.expm1(preds_high_log))
    
    df_results_high = df_test_high[['family', 'sales']].copy()
    df_results_high['prediction'] = preds_high_orig
    

    #  4. Xử lý và Train MÔ HÌNH 2 (Low-Volume) 
    
    print(f"Fold {fold}: Đang xử lý Mô hình 2 (Low-Volume)...")
    df_train_low[features_low] = df_train_low[features_low].fillna(0)
    df_test_low[features_low] = df_test_low[features_low].fillna(0)

    for col in categorical_features_low:
        all_categories = pd.concat([df_train_low[col], df_test_low[col]]).unique()
        df_train_low[col] = pd.Categorical(df_train_low[col], categories=all_categories)
        df_test_low[col] = pd.Categorical(df_test_low[col], categories=all_categories)

    X_train_low = df_train_low[features_low]
    y_train_low = df_train_low[target]
    X_test_low = df_test_low[features_low]
    y_test_low = df_test_low[target]

    xgb_model_2 = xgb.XGBRegressor(
        objective='reg:tweedie', tweedie_variance_power=1.5, eval_metric='rmse',
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        n_jobs=-1, random_state=42, early_stopping_rounds=100, enable_categorical=True
    )
    
    print(f"Fold {fold}: Đang train Mô hình 2 (Tweedie)...")
    xgb_model_2.fit(
        X_train_low, y_train_low,
        eval_set=[(X_test_low, y_test_low)],
        verbose=False
    )
    
    preds_low_orig = np.maximum(0, xgb_model_2.predict(X_test_low))
    
    df_results_low = df_test_low[['family', 'sales']].copy()
    df_results_low['prediction'] = preds_low_orig

    #  5. Gộp kết quả Fold này 
    df_results_fold = pd.concat([df_results_high, df_results_low])
    all_results_dfs.append(df_results_fold)
    
    print(f"Fold {fold} hoàn thành.")

print(" Cross-Validation Hoàn Tất ")

#  6. Tổng hợp và Đánh giá Kết quả CUỐI CÙNG 

df_results_FINAL = pd.concat(all_results_dfs)
df_results_FINAL_gt_zero = df_results_FINAL[df_results_FINAL['sales'] > 0]

print("\n 📊 Đánh giá Hiệu suất PIPELINE TỔNG HỢP (có Weights) ")

performance_data = []

all_families = df_results_FINAL['family'].unique()
print(f"Đang tính toán chỉ số cho toàn bộ {len(all_families)} nhóm hàng...")

for family in all_families:
    family_df = df_results_FINAL[df_results_FINAL['family'] == family]
    family_df_gt_zero = df_results_FINAL_gt_zero[df_results_FINAL_gt_zero['family'] == family]
    
    if len(family_df) == 0:
        continue

    family_mae = mean_absolute_error(family_df['sales'], family_df['prediction'])
    family_rmse = np.sqrt(mean_squared_error(family_df['sales'], family_df['prediction']))
    family_r2 = r2_score(family_df['sales'], family_df['prediction'])
    
    if len(family_df_gt_zero) > 0:
        family_mape = mean_absolute_percentage_error(family_df_gt_zero['sales'], family_df_gt_zero['prediction']) * 100
        family_accuracy = 100 - family_mape
    else:
        family_mape = np.nan
        family_accuracy = np.nan
    
    performance_data.append({
        'family_name': family,
        'MAE': family_mae, 
        'RMSE': family_rmse,
        'R2': family_r2,
        'MAPE': family_mape,
        'Accuracy (%)': family_accuracy,
        'Count': len(family_df)
    })

# 3. Tạo file CSV báo cáo (CHƯA CÓ DÒNG OVERALL)
performance_df = pd.DataFrame(performance_data)
# Sắp xếp 33 nhóm hàng theo Accuracy
performance_df = performance_df.sort_values(by='Accuracy (%)', ascending=False) 

# (THAY ĐỔI MỚI: TÍNH TOÁN VÀ THÊM DÒNG 'OVERALL')
print(" Đang tính toán 'Overall (5 Folds Average)' ")

# Tính toán trên toàn bộ data
overall_mae = mean_absolute_error(df_results_FINAL['sales'], df_results_FINAL['prediction'])
overall_rmse = np.sqrt(mean_squared_error(df_results_FINAL['sales'], df_results_FINAL['prediction']))
overall_r2 = r2_score(df_results_FINAL['sales'], df_results_FINAL['prediction'])

# Tính MAPE/Accuracy trên data > 0
overall_mape = mean_absolute_percentage_error(df_results_FINAL_gt_zero['sales'], df_results_FINAL_gt_zero['prediction']) * 100
overall_accuracy = 100 - overall_mape

# Lấy tổng count
overall_count = len(df_results_FINAL)

# Tạo dòng "Overall"
overall_row = pd.DataFrame([{
    'family_name': 'Overall (5 Folds Average)',
    'MAE': overall_mae,
    'RMSE': overall_rmse,
    'R2': overall_r2,
    'MAPE': overall_mape,
    'Accuracy (%)': overall_accuracy,
    'Count': overall_count
}])

# Gộp bảng (đã sắp xếp) với dòng "Overall" (sẽ nằm ở cuối)
performance_df_FINAL = pd.concat([performance_df, overall_row], ignore_index=True)

# Đổi tên file output
output_filename = 'xgb_models_FINAL_with_OVERALL_performance.csv'
performance_df_FINAL.to_csv(output_filename, index=False, float_format='%.4f')

print("" * 10)
print(f"\n Đã lưu vào Pipeline TỔNG HỢP vào file: {output_filename}")
total_time = time.time() - start_time
print(f"Tổng thời gian chạy: {total_time:.2f} giây.")

