import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import time

# 1. Cấu hình Chiến lược "Chia để trị"

# Các nhóm "Lõi" (Nhóm A)
HIGH_VOLUME_FAMILIES = [
    'GROCERY I', 'PRODUCE', 'BEVERAGES', 'DAIRY',
    'BREAD/BAKERY', 'DELI', 'EGGS'
]
N_FOLDS = 5
target = 'sales'

# Features cho Mô hình A (phức tạp)
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

# Features cho Mô hình B (đơn giản)
features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster', 
    'dcoilwtico', 'is_holiday', 'day_of_week', 'week_of_year', 'month', 'year', 
    'is_weekend'
]
categorical_features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster', 
    'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend'
]

# List để lưu kết quả của từng fold (Fold 1, Fold 2...)
fold_performance_data = [] 
# List để lưu TẤT CẢ dự đoán thô (để tính 33 nhóm và Overall)
all_results_dfs = []

print("Bắt đầu PIPELINE BÁO CÁO ĐẦY ĐỦ 2 PHẦN (Random Forest)...")
print("Cảnh báo: Đây là script nặng nhất và sẽ train rất lâu.")
start_time = time.time()

# 2. Bắt đầu Vòng lặp 5-Fold Cross-Validation

for fold in range(1, N_FOLDS + 1):
    print(f"\nĐang chạy Fold {fold}/{N_FOLDS}...")
    
    # Load data
    train_file = f'folds/train_fold_{fold}.csv'
    test_file = f'folds/test_fold_{fold}.csv'
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Tính "Trọng số" (Sample Weights)
    print(f"Fold {fold}: Đang tính toán Sample Weights...")
    family_sales_sum = df_train.groupby('family')['sales'].sum()
    family_weights_map = family_sales_sum / family_sales_sum.mean()
    
    # Tách data thành 2 nhóm (High và Low)
    print(f"Fold {fold}: Đang tách data (High/Low Volume)...")
    df_train_high = df_train[df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test_high = df_test[df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    
    df_train_low = df_train[~df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test_low = df_test[~df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()

    # 3. Xử lý và Train MÔ HÌNH A (High-Volume, RF)
    
    print(f"Fold {fold}: Đang xử lý Mô hình A (High-Volume, RF)...")
    df_train_high[features_high] = df_train_high[features_high].fillna(0)
    df_test_high[features_high] = df_test_high[features_high].fillna(0)
    
    # Gán trọng số TRƯỚC khi encode
    df_train_high['sample_weight'] = df_train_high['family'].map(family_weights_map)
    y_train_weights = df_train_high['sample_weight'].fillna(1.0) 
    
    # Chạy OrdinalEncoder
    encoder_high = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train_high[categorical_features_high] = encoder_high.fit_transform(df_train_high[categorical_features_high])
    df_test_high[categorical_features_high] = encoder_high.transform(df_test_high[categorical_features_high])
    
    X_train_high = df_train_high[features_high]
    X_test_high = df_test_high[features_high]
    
    # Kỹ thuật Log Transform
    y_train_high_log = np.log1p(df_train_high[target])
    y_test_high_orig = df_test_high[target] 
    
    # Khởi tạo Mô hình A (RF)
    rf_model_1 = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42,
        max_depth=10, min_samples_leaf=10 
    )
    
    print(f"Fold {fold}: Đang train Mô hình A (Log + Weights, RF)...")
    rf_model_1.fit(
        X_train_high, y_train_high_log,
        sample_weight=y_train_weights
    )
    
    # Dự đoán và chuyển ngược
    preds_high_log = rf_model_1.predict(X_test_high)
    preds_high_orig = np.maximum(0, np.expm1(preds_high_log)) 
    
    # Lưu kết quả Model A
    df_results_high = df_test_high[['family', 'sales']].copy()
    df_results_high['family'] = df_test[df_test['family'].isin(HIGH_VOLUME_FAMILIES)]['family']
    df_results_high['prediction'] = preds_high_orig
    
    # 4. Xử lý và Train MÔ HÌNH B (Low-Volume, Two-Part Model)
    
    print(f"Fold {fold}: Đang xử lý Mô hình B (Low-Volume, RF)...")
    df_train_low[features_low] = df_train_low[features_low].fillna(0)
    df_test_low[features_low] = df_test_low[features_low].fillna(0)

    # Dùng encoder riêng cho mô hình B
    encoder_low = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train_low[categorical_features_low] = encoder_low.fit_transform(df_train_low[categorical_features_low])
    df_test_low[categorical_features_low] = encoder_low.transform(df_test_low[categorical_features_low])

    # PHẦN 1 (Phân loại)
    print(f"Fold {fold}: Đang chuẩn bị & train Mô hình B-P1 (Phân loại)...")
    df_train_low['did_it_sell'] = (df_train_low[target] > 0).astype(int)
    X_train_class = df_train_low[features_low]
    y_train_class = df_train_low['did_it_sell']
    
    # Gỡ bỏ giới hạn max_depth để học sâu hơn
    rf_classifier = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    )
    rf_classifier.fit(X_train_class, y_train_class)

    # PHẦN 2 (Hồi quy)
    print(f"Fold {fold}: Đang chuẩn bị & train Mô hình B-P2 (Hồi quy)...")
    df_train_reg = df_train_low[df_train_low['did_it_sell'] == 1].copy()
    X_train_reg = df_train_reg[features_low]
    y_train_reg_log = np.log1p(df_train_reg[target])

    # Gỡ bỏ giới hạn max_depth để học sâu hơn
    rf_regressor = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42
    )
    # Chỉ train trên data sales > 0
    rf_regressor.fit(X_train_reg, y_train_reg_log)

    # PHẦN 3 (Kết hợp)
    print(f"Fold {fold}: Đang kết hợp dự đoán (P1 * P2)...")
    X_test_low = df_test_low[features_low]
    y_test_orig = df_test_low[target]
    
    # P1: Xác suất bán được hàng
    preds_prob = rf_classifier.predict_proba(X_test_low)[:, 1]
    # P2: Số lượng bán (nếu có)
    preds_reg_log = rf_regressor.predict(X_test_low)
    preds_reg_orig = np.maximum(0, np.expm1(preds_reg_log))
    
    # Dự đoán cuối cùng = (Xác suất P1) * (Số lượng P2)
    final_preds = preds_prob * preds_reg_orig
    
    # Lưu kết quả Model B
    df_results_low = df_test_low[['family', 'sales']].copy()
    df_results_low['family'] = df_test[~df_test['family'].isin(HIGH_VOLUME_FAMILIES)]['family']
    df_results_low['prediction'] = final_preds

    # 5. Gộp kết quả Fold này và Tính toán
    
    df_results_fold = pd.concat([df_results_high, df_results_low])
    
    fold_sales = df_results_fold['sales']
    fold_preds = df_results_fold['prediction']
    df_fold_gt_zero = df_results_fold[df_results_fold['sales'] > 0]
    
    fold_mae = mean_absolute_error(fold_sales, fold_preds)
    fold_rmse = np.sqrt(mean_squared_error(fold_sales, fold_preds))
    fold_r2 = r2_score(fold_sales, fold_preds)
    
    if len(df_fold_gt_zero) > 0:
        fold_mape = mean_absolute_percentage_error(df_fold_gt_zero['sales'], df_fold_gt_zero['prediction']) * 100
        fold_accuracy = 100 - fold_mape
    else:
        fold_mape = np.nan
        fold_accuracy = np.nan

    # Lưu kết quả tóm tắt của Fold này
    fold_performance_data.append({
        'family_name': f'Fold {fold}', 
        'MAE': fold_mae, 'RMSE': fold_rmse, 'R2': fold_r2,
        'MAPE': fold_mape, 'Accuracy (%)': fold_accuracy, 'Count': len(df_results_fold)
    })
    
    # Lưu kết quả thô
    all_results_dfs.append(df_results_fold)
    
    print(f"Fold {fold} hoàn thành.")

print("\nCross-Validation Hoàn Tất. Bắt đầu tổng hợp báo cáo...")

# 6. Tổng hợp Báo cáo cuối cùng (2 Phần)

df_results_FINAL = pd.concat(all_results_dfs)
df_results_FINAL_gt_zero = df_results_FINAL[df_results_FINAL['sales'] > 0]

# PHẦN 1: BÁO CÁO CHI TIẾT (33 NHÓM HÀNG)
print("Đang tính toán Phần 1 (33 Nhóm hàng)...")
family_performance_data = []
all_families = df_results_FINAL['family'].unique()

for family in all_families:
    family_df = df_results_FINAL[df_results_FINAL['family'] == family]
    family_df_gt_zero = df_results_FINAL_gt_zero[df_results_FINAL_gt_zero['family'] == family]
    if len(family_df) == 0: continue

    family_mae = mean_absolute_error(family_df['sales'], family_df['prediction'])
    family_rmse = np.sqrt(mean_squared_error(family_df['sales'], family_df['prediction']))
    family_r2 = r2_score(family_df['sales'], family_df['prediction'])
    
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

# Sắp xếp 33 nhóm
performance_df_families = pd.DataFrame(family_performance_data)
performance_df_families = performance_df_families.sort_values(by='R2', ascending=False) 

# PHẦN 2: BÁO CÁO TÓM TẮT (5 FOLDS + OVERALL)
print("Đang tính toán Phần 2 (5 Folds + Overall)...")

# Lấy bảng 5 Folds
performance_df_folds = pd.DataFrame(fold_performance_data)

# Tính dòng 'Overall'
overall_mae = mean_absolute_error(df_results_FINAL['sales'], df_results_FINAL['prediction'])
overall_rmse = np.sqrt(mean_squared_error(df_results_FINAL['sales'], df_results_FINAL['prediction']))
overall_r2 = r2_score(df_results_FINAL['sales'], df_results_FINAL['prediction'])
overall_mape = mean_absolute_percentage_error(df_results_FINAL_gt_zero['sales'], df_results_FINAL_gt_zero['prediction']) * 100
overall_accuracy = 100 - overall_mape
overall_count = len(df_results_FINAL)

overall_row = pd.DataFrame([{
    'family_name': 'Overall (5 Folds Average)',
    'MAE': overall_mae, 'RMSE': overall_rmse, 'R2': overall_r2,
    'MAPE': overall_mape, 'Accuracy (%)': overall_accuracy, 'Count': overall_count
}])

# Gộp 5 Folds và Overall
performance_df_summary = pd.concat([performance_df_folds, overall_row], ignore_index=True)

# PHẦN 3: GỘP 2 BÁO CÁO LẠI THÀNH FILE CUỐI CÙNG
print("Đang gộp 2 phần báo cáo...")
# Gộp Phần 1 (33 nhóm) và Phần 2 (6 dòng tóm tắt)
performance_df_FINAL_out = pd.concat([performance_df_families, performance_df_summary], ignore_index=True)

output_filename = 'rf_models_COMPLETE_REPORT.csv'
performance_df_FINAL_out.to_csv(output_filename, index=False, float_format='%.4f')

print(f"\n✅ HOÀN THÀNH! Đã lưu BÁO CÁO ĐẦY ĐỦ 2 PHẦN (Random Forest) vào file: {output_filename}")
total_time = time.time() - start_time
print(f"Tổng thời gian chạy: {total_time:.2f} giây.")