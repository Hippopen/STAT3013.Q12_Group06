import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import time

# 1. Cấu hình Mô hình A (Nhóm Lõi/Doanh số cao)

# Các nhóm "Lõi" để train
HIGH_VOLUME_FAMILIES = [
    'GROCERY I', 'PRODUCE', 'BEVERAGES', 'DAIRY',
    'BREAD/BAKERY', 'DELI', 'EGGS'
]
N_FOLDS = 5
target = 'sales'

# Features cho Mô hình A (bao gồm lag, rolling)
features_high = [
    'store_nbr', 'family', 'onpromotion', 'transactions', 'city', 'state', 
    'type', 'cluster', 'dcoilwtico', 'is_holiday', 'day_of_week', 
    'week_of_year', 'month', 'year', 'is_weekend', 'sales_lag_7', 
    'sales_lag_14', 'rolling_mean_30'
]
categorical_features_high = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster', 
    'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend'
]

# List để lưu kết quả
all_original_sales = []
all_original_preds = []
all_original_family = []

print(f"Bắt đầu xây dựng Mô hình A (Random Forest - Nhóm Lõi)...")
print(f"Train cho {len(HIGH_VOLUME_FAMILIES)} nhóm hàng 'Biến động cao'.")
print("Cảnh báo: Quá trình này sẽ mất nhiều thời gian.")
start_time = time.time()

# 2. Bắt đầu Vòng lặp 5-Fold Cross-Validation

for fold in range(1, N_FOLDS + 1):
    print(f"\nĐang chạy Fold {fold}/{N_FOLDS}...")
    
    # Load data
    train_file = f'folds/train_fold_{fold}.csv'
    test_file = f'folds/test_fold_{fold}.csv'
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # LỌC DATA (Chỉ giữ lại Nhóm A)
    df_train_high = df_train[df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test_high = df_test[df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()

    # Tính "Trọng số" (Sample Weights)
    family_sales_sum = df_train_high.groupby('family')['sales'].sum()
    family_weights_map = family_sales_sum / family_sales_sum.mean()
    
    # 3. Xử lý và Train MÔ HÌNH A
    
    df_train_high[features_high] = df_train_high[features_high].fillna(0)
    df_test_high[features_high] = df_test_high[features_high].fillna(0)
    
    # SỬA LỖI NAN: Gán trọng số TRƯỚC khi encode
    df_train_high['sample_weight'] = df_train_high['family'].map(family_weights_map)
    y_train_weights = df_train_high['sample_weight'].fillna(1.0) # Dùng 1.0 (trung tính) nếu có lỗi
    
    # Chạy OrdinalEncoder để chuyển category sang số
    encoder_high = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train_high[categorical_features_high] = encoder_high.fit_transform(df_train_high[categorical_features_high])
    df_test_high[categorical_features_high] = encoder_high.transform(df_test_high[categorical_features_high])
    
    # Chuẩn bị X (features) và y (target)
    X_train_high = df_train_high[features_high]
    X_test_high = df_test_high[features_high]
    
    # Kỹ thuật Log Transform: dự đoán log(sales + 1)
    y_train_high_log = np.log1p(df_train_high[target])
    y_test_high_orig = df_test_high[target] # Giữ y gốc để so sánh
    
    # Khởi tạo Mô hình A (RF)
    rf_model_1 = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42,
        max_depth=10, min_samples_leaf=10 
    )
    
    print(f"Fold {fold}: Đang train Mô hình A (Log + Weights, RF)...")
    
    # Train mô hình
    rf_model_1.fit(
        X_train_high, y_train_high_log,
        sample_weight=y_train_weights # Thêm trọng số
    )
    
    # Dự đoán (trên thang Log)
    preds_high_log = rf_model_1.predict(X_test_high)
    # Chuyển ngược về thang đo gốc (dùng expm1 để đảo ngược log1p)
    preds_high_orig = np.maximum(0, np.expm1(preds_high_log)) 
    
    # Lưu kết quả ở thang đo gốc
    all_original_sales.append(y_test_high_orig)
    all_original_preds.append(pd.Series(preds_high_orig, index=y_test_high_orig.index))
    # Lấy lại tên 'family' gốc (vì đã bị encode thành số)
    all_original_family.append(df_test[df_test['family'].isin(HIGH_VOLUME_FAMILIES)]['family'])
    
    print(f"Fold {fold} hoàn thành.")

print("\nCross-Validation Hoàn Tất. Bắt đầu tổng hợp báo cáo cho Mô hình A...")

# 4. Tổng hợp Báo cáo cuối cùng (chỉ cho Nhóm A)

df_results_FINAL = pd.DataFrame({
    'family': pd.concat(all_original_family),
    'sales': pd.concat(all_original_sales),
    'prediction': pd.concat(all_original_preds)
})

# PHẦN 1: BÁO CÁO CHI TIẾT (7 NHÓM HÀNG)
print("Đang tính toán Phần 1 (Chi tiết 7 Nhóm)...")
family_performance_data = []

for family in HIGH_VOLUME_FAMILIES:
    family_df = df_results_FINAL[df_results_FINAL['family'] == family]
    if len(family_df) == 0: continue

    family_mae = mean_absolute_error(family_df['sales'], family_df['prediction'])
    family_rmse = np.sqrt(mean_squared_error(family_df['sales'], family_df['prediction']))
    family_r2 = r2_score(family_df['sales'], family_df['prediction'])
    
    family_df_gt_zero = family_df[family_df['sales'] > 0]
    if len(family_df_gt_zero) > 0:
        family_mape = mean_absolute_percentage_error(family_df_gt_zero['sales'], family_df_gt_zero['prediction']) * 100
        family_accuracy = 100 - family_mape
    else:
        family_mape = np.nan
        family_accuracy = np.nan
    
    family_performance_data.append({
        'family_name': family,
        'MAE': family_mae, 'RMSE': family_rmse, 'R2': family_r2,
        'MAPE': family_mape, 'Accuracy (%)': family_accuracy, 'Count': len(family_df)
    })

performance_df_families = pd.DataFrame(family_performance_data)
performance_df_families = performance_df_families.sort_values(by='Accuracy (%)', ascending=False) 

# PHẦN 2: BÁO CÁO TÓM TẮT (OVERALL CỦA NHÓM A)
print("Đang tính toán Phần 2 (Overall Nhóm A)...")

df_results_FINAL_gt_zero = df_results_FINAL[df_results_FINAL['sales'] > 0]
overall_mae = mean_absolute_error(df_results_FINAL['sales'], df_results_FINAL['prediction'])
overall_rmse = np.sqrt(mean_squared_error(df_results_FINAL['sales'], df_results_FINAL['prediction']))
overall_r2 = r2_score(df_results_FINAL['sales'], df_results_FINAL['prediction'])
overall_mape = mean_absolute_percentage_error(df_results_FINAL_gt_zero['sales'], df_results_FINAL_gt_zero['prediction']) * 100
overall_accuracy = 100 - overall_mape
overall_count = len(df_results_FINAL)

overall_row = pd.DataFrame([{
    'family_name': 'Overall (Nhóm A)',
    'MAE': overall_mae, 'RMSE': overall_rmse, 'R2': overall_r2,
    'MAPE': overall_mape, 'Accuracy (%)': overall_accuracy, 'Count': overall_count
}])

# PHẦN 3: GỘP 2 BÁO CÁO LẠI
print("Đang gộp 2 phần báo cáo...")
performance_df_FINAL_out = pd.concat([performance_df_families, overall_row], ignore_index=True)

output_filename = 'rf_MODEL_A_high_performance.csv'
performance_df_FINAL_out.to_csv(output_filename, index=False, float_format='%.4f')

print(f"\n✅ HOÀN THÀNH (Mô hình A)! Đã lưu báo cáo (chỉ Nhóm A) vào file: {output_filename}")
total_time = time.time() - start_time
print(f"Tổng thời gian chạy: {total_time:.2f} giây.")