import pandas as pd
import numpy as np
import shap
import joblib
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import os
from pandas.api.types import is_object_dtype, is_categorical_dtype

# ----- 1. CẤU HÌNH (Chỉnh lại nếu cần) -----
# Chọn 1 fold để phân tích (ví dụ: fold 5)
FOLD_NAME = "fold_5"

# Đường dẫn đến model và file test (phải khớp với FOLD_NAME)
MODEL_FILE_PKL = os.path.join("models", f"lightgbm_{FOLD_NAME}.pkl")
TEST_FILE = os.path.join("results", f"test_{FOLD_NAME}.csv") # Dùng file test_... gốc

# Thư mục lưu kết quả SHAP
SHAP_OUTPUT_DIR = "results_shap_lightgbm"
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# ----- 2. CÀI ĐẶT FEATURE (Sao chép y hệt file train) -----
TARGET_COL = "sales"
DATE_COL = "date"
EXCLUDE_COLS = [TARGET_COL, DATE_COL]

# ----- 3. CẢNH BÁO TỐC ĐỘ -----
SAMPLE_SIZE = 5000 # Lấy mẫu 5000 dòng để chạy nhanh

print(f"--- Bắt đầu phân tích SHAP cho LightGBM {FOLD_NAME} ---")

# ----- 4. Tải Model -----
print(f"Đang tải model: {MODEL_FILE_PKL}")
try:
    model = joblib.load(MODEL_FILE_PKL)
except Exception as e:
    print(f"LỖI: Không thể tải model. Bạn chắc chắn file '{MODEL_FILE_PKL}' tồn tại?")
    print(f"Lỗi chi tiết: {e}")
    exit()

# ----- 5. Tải và Chuẩn bị Dữ liệu Test -----
print(f"Đang tải dữ liệu test: {TEST_FILE}")
try:
    test_df = pd.read_csv(TEST_FILE, parse_dates=[DATE_COL])
except Exception as e:
    print(f"LỖI: Không thể tải file test. Bạn chắc chắn file '{TEST_FILE}' tồn tại?")
    print(f"Lỗi chi tiết: {e}")
    exit()

# ----- BƯỚC QUAN TRỌNG: TÁI TẠO PREPROCESSING -----
# Chúng ta phải tái tạo y hệt các bước xử lý trong file train
print("Tái tạo lại features và dtypes...")

# 1. Xác định feature_cols (loại trừ target, date)
feature_cols = [c for c in test_df.columns if c not in EXCLUDE_COLS]

# 2. Xác định và convert categorical features
# (Dùng logic y hệt file train)
cat_feature_cols = [c for c in feature_cols if is_object_dtype(test_df[c]) or is_categorical_dtype(test_df[c])]
print(f"Tìm thấy {len(cat_feature_cols)} categorical features. Đang convert...")
for c in cat_feature_cols:
    # Điền 'missing' và ép kiểu về 'category'
    test_df[c] = pd.Categorical(test_df[c].fillna('missing'))

# 3. Xác định và fillna cho numeric features
num_feature_cols = [c for c in feature_cols if c not in cat_feature_cols]
print(f"Tìm thấy {len(num_feature_cols)} numeric features. Đang fillna(0)...")
if len(num_feature_cols) > 0:
    test_df[num_feature_cols] = test_df[num_feature_cols].fillna(0)

# 4. Tạo X_test (chỉ chứa các cột feature đã xử lý)
X_test = test_df[feature_cols]
print("Tái tạo Dtypes hoàn tất.")

# ----- 6. Lấy mẫu (Sampling) -----
if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(X_test):
    print(f"Lấy mẫu {SAMPLE_SIZE} dòng từ X_test để tính SHAP...")
    X_sample = X_test.sample(n=SAMPLE_SIZE, random_state=42)
else:
    print(f"Sử dụng toàn bộ {len(X_test)} dòng để tính SHAP...")
    X_sample = X_test

# ----- 7. Tính toán SHAP -----
print("Khởi tạo SHAP TreeExplainer...")
# LightGBM có thể gặp vấn đề nếu dtypes không khớp. 
# Việc convert 'category' ở trên là rất quan trọng.
explainer = shap.TreeExplainer(model)

print(f"Đang tính toán SHAP values cho {len(X_sample)} dòng. Xin chờ...")
shap_values = explainer.shap_values(X_sample)
print("Tính SHAP values hoàn tất.")

# ----- 8. Vẽ và Lưu Biểu đồ -----
# (Đổi tên file output thành 'lightgbm_...')

# A. SUMMARY PLOT (DOT)
print("Đang vẽ Summary Plot (dot)...")
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
summary_dot_path = os.path.join(SHAP_OUTPUT_DIR, f"lightgbm_{FOLD_NAME}_summary_dot.png")
plt.savefig(summary_dot_path, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {summary_dot_path}")

# B. SUMMARY PLOT (BAR)
print("Đang vẽ Summary Plot (bar)...")
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
summary_bar_path = os.path.join(SHAP_OUTPUT_DIR, f"lightgbm_{FOLD_NAME}_summary_bar.png")
plt.savefig(summary_bar_path, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {summary_bar_path}")

# C. DEPENDENCE PLOTS (Top 5 features)
print("Đang vẽ Dependence Plots cho top features...")
try:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_sample.columns
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': mean_abs_shap})
    df_imp = df_imp.sort_values('importance', ascending=False)
    TOP_N_FEATURES = df_imp['feature'].head(5).tolist()
    print(f"Top 5 features: {TOP_N_FEATURES}")
except Exception:
    TOP_N_FEATURES = ['lag_7', 'rolling_mean_30', 'onpromotion'] # Dự phòng

for feature in TOP_N_FEATURES:
    if feature in X_sample.columns:
        print(f"  - Đang vẽ cho: {feature}")
        plt.figure()
        try:
            shap.dependence_plot(feature, shap_values, X_sample, show=False)
            dep_path = os.path.join(SHAP_OUTPUT_DIR, f"lightgbm_{FOLD_NAME}_dependence_{feature}.png")
            plt.savefig(dep_path, bbox_inches='tight')
            plt.close()
            print(f"    Đã lưu: {dep_path}")
        except Exception as e:
            print(f"    Lỗi khi vẽ dependence plot cho '{feature}': {e}")
            plt.close()

print("\n--- Hoàn tất! ---")
print(f"Tất cả hình ảnh SHAP đã được lưu vào thư mục: {SHAP_OUTPUT_DIR}")