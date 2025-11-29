import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import time

#  1. Cài đặt Cấu hình (Mô hình 1: Nhóm Biến động cao) 

HIGH_VOLUME_FAMILIES = [
    'GROCERY I',
    'PRODUCE',
    'BEVERAGES',
    'DAIRY',
    'BREAD/BAKERY',
    'DELI',
    'EGGS'
]

N_FOLDS = 5

features = [
    'store_nbr', 'family', 'onpromotion', 'transactions', 
    'city', 'state', 'type', 'cluster', 'dcoilwtico', 
    'is_holiday', 'day_of_week', 'week_of_year', 'month', 'year', 
    'is_weekend', 'sales_lag_7', 'sales_lag_14', 'rolling_mean_30'
]
target = 'sales'

categorical_features = [
    'store_nbr', 'family', 'city', 'state', 'type', 
    'cluster', 'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend'
]

all_original_sales = []
all_original_preds = []
all_original_family = []

print(f"Bắt đầu Kế hoạch 'Chia để trị' (Mô hình 1 - Sửa lỗi Log Transform)...")
print(f"Train cho {len(HIGH_VOLUME_FAMILIES)} nhóm 'Biến động cao' (trên thang Log).")
start_time = time.time()

#  2. Chạy Vòng lặp 5-Fold Cross-Validation 

for fold in range(1, N_FOLDS + 1):
    print(f" Đang chạy Fold {fold}/{N_FOLDS} ")
    
    train_file = f'folds/train_fold_{fold}.csv'
    test_file = f'folds/test_fold_{fold}.csv'
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    df_train = df_train[df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test = df_test[df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    
    df_train[features] = df_train[features].fillna(0)
    df_test[features] = df_test[features].fillna(0)
    
    print(f"Fold {fold}: Đang xử lý categorical features...")
    for col in categorical_features:
        all_categories = pd.concat([df_train[col], df_test[col]]).unique()
        df_train[col] = pd.Categorical(df_train[col], categories=all_categories)
        df_test[col] = pd.Categorical(df_test[col], categories=all_categories)

    X_train = df_train[features]
    X_test = df_test[features]
    
    y_train_log = np.log1p(df_train[target])
    y_test_log = np.log1p(df_test[target])
    y_test_orig = df_test[target]

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=100,
        enable_categorical=True
    )
    
    print(f"Fold {fold}: Bắt đầu training XGBoost (trên thang Log)...")
    xgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_test, y_test_log)],
        verbose=False
    )
    print(f"Fold {fold}: Training hoàn tất. Bắt đầu dự đoán...")
    
    preds_log = xgb_model.predict(X_test)
    
    preds_orig = np.maximum(0, np.expm1(preds_log))
    
    all_original_sales.append(y_test_orig)
    all_original_preds.append(pd.Series(preds_orig, index=y_test_orig.index))
    all_original_family.append(df_test['family'])
    
    print(f"Fold {fold} hoàn thành.")

print(" Cross-Validation Hoàn Tất ")

#  3. Tổng hợp và Đánh giá Kết quả 

df_results = pd.DataFrame({
    'family': pd.concat(all_original_family),
    'sales': pd.concat(all_original_sales),
    'prediction': pd.concat(all_original_preds)
})

print("\n 📊 Đánh giá Hiệu suất (Mô hình 1 - Đã Log Transform) ")

performance_data = []

for family in HIGH_VOLUME_FAMILIES:
    family_df = df_results[df_results['family'] == family]
    
    if len(family_df) == 0:
        continue

    family_mae = mean_absolute_error(family_df['sales'], family_df['prediction'])
    family_rmse = np.sqrt(mean_squared_error(family_df['sales'], family_df['prediction']))
    family_r2 = r2_score(family_df['sales'], family_df['prediction'])
    
    family_df_gt_zero = family_df[family_df['sales'] > 0]
    family_mape = mean_absolute_percentage_error(family_df_gt_zero['sales'], family_df_gt_zero['prediction']) * 100
    # Định nghĩa biến accuracy
    family_accuracy = 100 - family_mape
    
    performance_data.append({
        'family_name': family,
        'MAE': family_mae, 
        'RMSE': family_rmse,
        'R2': family_r2,
        'MAPE': family_mape,
        # (ĐÂY LÀ DÒNG ĐÃ SỬA LỖI)
        # Sửa từ 'family_mape' thành 'family_accuracy'
        'Accuracy (%)': family_accuracy, 
        'Count': len(family_df)
    })
    
    print(f"\nNhóm hàng: {family}")
    print(f"  MAE (Sai số tuyệt đối): {family_mae:.4f}")
    print(f"  RMSE: {family_rmse:.4f}")
    print(f"  R2 (R-squared): {family_r2:.4f}")
    print(f"  MAPE: {family_mape:.4f}%")
    print(f"  Accuracy: {family_accuracy:.4f}%") # < Sẽ in ra số đúng

# 3. Tạo file CSV báo cáo
performance_df = pd.DataFrame(performance_data)
performance_df = performance_df.sort_values(by='Accuracy (%)', ascending=False) 

# Ghi đè file cũ
output_filename = 'xgb_model_1_high_volume_LOG_performance.csv'
performance_df.to_csv(output_filename, index=False, float_format='%.4f')

print("" * 10)
print(f"\n Đã lưu Mô hình 1 vào file: {output_filename}")
total_time = time.time() - start_time
print(f"Tổng thời gian chạy: {total_time:.2f} giây.")

