import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import time

# 1. Cấu hình Mô hình B (Nhóm Rời rạc/Doanh số thấp)

# Các nhóm "Lõi" (Nhóm A) để LOẠI TRỪ
HIGH_VOLUME_FAMILIES = [
    'GROCERY I', 'PRODUCE', 'BEVERAGES', 'DAIRY',
    'BREAD/BAKERY', 'DELI', 'EGGS'
]
N_FOLDS = 5
target = 'sales'

# Features cho Mô hình B (bộ features đơn giản)
features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster', 
    'dcoilwtico', 'is_holiday', 'day_of_week', 'week_of_year', 'month', 'year', 
    'is_weekend'
]
categorical_features_low = [
    'store_nbr', 'family', 'city', 'state', 'type', 'cluster', 
    'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend'
]

# List để lưu kết quả
all_original_sales = []
all_original_preds = []
all_original_family = []

print(f"Bắt đầu xây dựng Mô hình B (Random Forest - Sửa lỗi)...")
print(f"Áp dụng 'Mô hình 2-phần' (Gỡ bỏ giới hạn max_depth).")
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

    # LỌC DATA (Chỉ giữ lại Nhóm B)
    df_train_low = df_train[~df_train['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    df_test_low = df_test[~df_test['family'].isin(HIGH_VOLUME_FAMILIES)].copy()
    
    # 3. Xử lý và Train MÔ HÌNH B (Two-Part Model)
    
    print(f"Fold {fold}: Đang xử lý features (FillNA & Encoder)...")
    df_train_low[features_low] = df_train_low[features_low].fillna(0)
    df_test_low[features_low] = df_test_low[features_low].fillna(0)
    
    encoder_low = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_train_low[categorical_features_low] = encoder_low.fit_transform(df_train_low[categorical_features_low])
    df_test_low[categorical_features_low] = encoder_low.transform(df_test_low[categorical_features_low])
    
    # PHẦN 1: MÔ HÌNH PHÂN LOẠI (Classification)
    
    print(f"Fold {fold}: Đang chuẩn bị Mô hình P1 (Phân loại)...")
    df_train_low['did_it_sell'] = (df_train_low[target] > 0).astype(int)
    
    X_train_class = df_train_low[features_low]
    y_train_class = df_train_low['did_it_sell']
    
    # Khởi tạo mô hình Classifier (KHÔNG GIỚI HẠN max_depth)
    rf_classifier = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
        # Đã gỡ bỏ max_depth và min_samples_leaf
    )
    
    print(f"Fold {fold}: Đang train Mô hình P1 (Phân loại)...")
    rf_classifier.fit(X_train_class, y_train_class)

    # PHẦN 2: MÔ HÌNH HỒI QUY (Regression)
    
    print(f"Fold {fold}: Đang chuẩn bị Mô hình P2 (Hồi quy)...")
    # Chỉ train trên data CÓ BÁN HÀNG (sales > 0)
    df_train_reg = df_train_low[df_train_low['did_it_sell'] == 1].copy()
    
    X_train_reg = df_train_reg[features_low]
    y_train_reg_log = np.log1p(df_train_reg[target])

    # Khởi tạo mô hình Regressor (KHÔNG GIỚI HẠN max_depth)
    rf_regressor = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42
        # Đã gỡ bỏ max_depth và min_samples_leaf
    )
    
    print(f"Fold {fold}: Đang train Mô hình P2 (Hồi quy)...")
    rf_regressor.fit(X_train_reg, y_train_reg_log)

    # PHẦN 3: KẾT HỢP DỰ ĐOÁN
    
    print(f"Fold {fold}: Đang kết hợp dự đoán (P1 * P2)...")
    X_test_low = df_test_low[features_low]
    y_test_orig = df_test_low[target] # y gốc để so sánh
    
    # P1: Xác suất bán được hàng
    preds_prob = rf_classifier.predict_proba(X_test_low)[:, 1]
    
    # P2: Số lượng bán (nếu có), trên thang Log
    preds_reg_log = rf_regressor.predict(X_test_low)
    preds_reg_orig = np.maximum(0, np.expm1(preds_reg_log))
    
    # Dự đoán cuối cùng = (Xác suất P1) * (Số lượng P2)
    final_preds = preds_prob * preds_reg_orig
    
    # Lưu kết quả
    all_original_sales.append(y_test_orig)
    all_original_preds.append(pd.Series(final_preds, index=y_test_orig.index))
    all_original_family.append(df_test[~df_test['family'].isin(HIGH_VOLUME_FAMILIES)]['family'])
    
    print(f"Fold {fold} hoàn thành.")

print("\nCross-Validation Hoàn Tất. Bắt đầu tổng hợp báo cáo cho Mô hình B...")

# 4. Tổng hợp Báo cáo cuối cùng (chỉ cho Nhóm B)

df_results_FINAL = pd.DataFrame({
    'family': pd.concat(all_original_family),
    'sales': pd.concat(all_original_sales),
    'prediction': pd.concat(all_original_preds)
})

# PHẦN 1: BÁO CÁO CHI TIẾT (26 NHÓM HÀNG CÒN LẠI)
print("Đang tính toán Phần 1 (Chi tiết 26 Nhóm)...")
family_performance_data = []

ALL_LOW_VOLUME_FAMILIES = df_results_FINAL['family'].unique()

for family in ALL_LOW_VOLUME_FAMILIES:
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
performance_df_families = performance_df_families.sort_values(by='R2', ascending=False) 

# PHẦN 2: BÁO CÁO TÓM TẮT (OVERALL CỦA NHÓM B)
print("Đang tính toán Phần 2 (Overall Nhóm B)...")

df_results_FINAL_gt_zero = df_results_FINAL[df_results_FINAL['sales'] > 0]
overall_mae = mean_absolute_error(df_results_FINAL['sales'], df_results_FINAL['prediction'])
overall_rmse = np.sqrt(mean_squared_error(df_results_FINAL['sales'], df_results_FINAL['prediction']))
overall_r2 = r2_score(df_results_FINAL['sales'], df_results_FINAL['prediction'])
overall_mape = mean_absolute_percentage_error(df_results_FINAL_gt_zero['sales'], df_results_FINAL_gt_zero['prediction']) * 100
overall_accuracy = 100 - overall_mape
overall_count = len(df_results_FINAL)

overall_row = pd.DataFrame([{
    'family_name': 'Overall (Nhóm B - Two-Part Model)',
    'MAE': overall_mae, 'RMSE': overall_rmse, 'R2': overall_r2,
    'MAPE': overall_mape, 'Accuracy (%)': overall_accuracy, 'Count': overall_count
}])

# PHẦN 3: GỘP 2 BÁO CÁO LẠI
print("Đang gộp 2 phần báo cáo...")
performance_df_FINAL_out = pd.concat([performance_df_families, overall_row], ignore_index=True)

# Đổi tên file để không ghi đè
output_filename = 'rf_MODEL_B_low_performance_v2.csv'
performance_df_FINAL_out.to_csv(output_filename, index=False, float_format='%.4f')

print(f"\n✅ HOÀN THÀNH (Mô hình B - Sửa lỗi)! Đã lưu báo cáo vào file: {output_filename}")
total_time = time.time() - start_time
print(f"Tổng thời gian chạy: {total_time:.2f} giây.")