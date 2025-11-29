import pandas as pd
import numpy as np
import xgboost as xgb
# (THÊM MAE)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error 
import time

#  1. Cài đặt Cấu hình (Mô hình 2: Nhóm Ổn định) 

HIGH_VOLUME_FAMILIES = [
    'GROCERY I', 'PRODUCE', 'BEVERAGES', 'DAIRY',
    'BREAD/BAKERY', 'DELI', 'EGGS'
]

N_FOLDS = 5

features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster', 
    'dcoilwtico', 'is_holiday', 'day_of_week', 'week_of_year', 'month', 'year', 
    'is_weekend'
]
target = 'sales'

categorical_features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 
    'cluster', 'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend'
]

all_results_dfs = []

print(f"Bắt đầu Kế hoạch 'Chia để trị' (Mô hình 2 - Sửa lỗi)...")
print(f"Train cho nhóm 'Ổn định' (dùng mục tiêu Tweedie).")
start_time = time.time()

#  2. Chạy Vòng lặp 5-Fold Cross-Validation 

for fold in range(1, N_FOLDS + 1):
    print(f" Đang chạy Fold {fold}/{N_FOLDS} ")
    
    train_file = f'folds/train_fold_{fold}.csv'
    test_file = f'folds/test_fold_{fold}.csv'
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Lọc data: Loại trừ 7 nhóm cao
    df_train = df_train[~df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test = df_test[~df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    
    df_train[features_low] = df_train[features_low].fillna(0)
    df_test[features_low] = df_test[features_low].fillna(0)
    
    print(f"Fold {fold}: Đang xử lý categorical features...")
    for col in categorical_features_low:
        all_categories = pd.concat([df_train[col], df_test[col]]).unique()
        df_train[col] = pd.Categorical(df_train[col], categories=all_categories)
        df_test[col] = pd.Categorical(df_test[col], categories=all_categories)

    X_train = df_train[features_low]
    y_train = df_train[target]
    
    X_test = df_test[features_low]
    y_test = df_test[target]
    
    # 6. Khởi tạo và Huấn luyện (THAY ĐỔI QUAN TRỌNG)
    # Sử dụng 'reg:tweedie' thay vì 'reg:squarederror'
    xgb_model_low = xgb.XGBRegressor(
        objective='reg:tweedie',    # < THAY ĐỔI
        tweedie_variance_power=1.5, # (tham số cho Tweedie, 1.5 là điểm khởi đầu tốt)
        eval_metric='rmse',         # (Vẫn theo dõi RMSE, nhưng mục tiêu là Tweedie)
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=100,
        enable_categorical=True
    )
    
    print(f"Fold {fold}: Bắt đầu training XGBoost (Tweedie)...")
    xgb_model_low.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print(f"Fold {fold}: Training hoàn tất. Bắt đầu dự đoán...")
    
    preds = xgb_model_low.predict(X_test)
    
    df_test_results = df_test[['family', 'sales']].copy()
    df_test_results['prediction'] = preds
    all_results_dfs.append(df_test_results)
    
    print(f"Fold {fold} hoàn thành.")

print(" Cross-Validation Hoàn Tất ")

#  3. Tổng hợp và Đánh giá Kết quả 

all_results = pd.concat(all_results_dfs)
all_results['prediction'] = all_results['prediction'].apply(lambda x: max(0, x))
all_results_gt_zero = all_results[all_results['sales'] > 0]

print("\n 📊 Đánh giá Hiệu suất (Mô hình 2 - Dùng Tweedie) ")

performance_data = []

all_families_low = all_results['family'].unique()
print(f"Đang tính toán chỉ số (có MAE) cho {len(all_families_low)} nhóm hàng còn lại...")

for family in all_families_low:
    family_df = all_results[all_results['family'] == family]
    family_df_gt_zero = all_results_gt_zero[all_results_gt_zero['family'] == family]
    
    if len(family_df) == 0:
        continue

    # (THÊM MAE)
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
        'MAE': family_mae, # < THÊM
        'RMSE': family_rmse,
        'R2': family_r2,
        'MAPE': family_mape,
        'Accuracy (%)': family_accuracy,
        'Count': len(family_df)
    })

# 3. Tạo file CSV báo cáo
performance_df = pd.DataFrame(performance_data)
# Sắp xếp theo R2 (chỉ số quan trọng nhất cho nhóm này)
performance_df = performance_df.sort_values(by='R2', ascending=False) 

output_filename = 'xgb_model_2_low_volume_TWEEDIE_performance.csv'
performance_df.to_csv(output_filename, index=False, float_format='%.4f')

print("" * 10)
print(f"\n Đã lưu Mô hình 2 vào file: {output_filename}")
total_time = time.time() - start_time
print(f"Tổng thời gian chạy: {total_time:.2f} giây.")

