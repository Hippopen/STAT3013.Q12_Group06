import pandas as pd
import numpy as np
import shap
import joblib
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import os

# ----- 1. CẤU HÌNH (Bạn PHẢI chỉnh lại cho đúng) -----
# Chọn 1 fold để phân tích (ví dụ: fold 5)
FOLD_NAME = "fold_5"

# Đường dẫn đến model và file test (phải khớp với FOLD_NAME)
MODEL_FILE = os.path.join("models_catboost", f"catboost_{FOLD_NAME}.cbm")
TEST_FILE = os.path.join("results_catboost", f"test_{FOLD_NAME}.csv")

# Thư mục lưu kết quả SHAP
SHAP_OUTPUT_DIR = "results_shap_catboost"
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# ----- 2. CÀI ĐẶT FEATURE (Sao chép y hệt file train) -----
# Các cột này PHẢI GIỐNG HỆT file train
TARGET_COL = "sales"
DATE_COL = "date"
EXCLUDE_COLS = [TARGET_COL, DATE_COL]

# ----- 3. CẢNH BÁO TỐC ĐỘ -----
# Tính SHAP cho hàng chục ngàn dòng CỰC KỲ CHẬM (có thể mất vài giờ).
# Chúng ta sẽ lấy mẫu (sample) để chạy nhanh hơn và kết quả vẫn đáng tin cậy.
# Nếu bạn muốn chạy trên toàn bộ dữ liệu, đặt SAMPLE_SIZE = None
SAMPLE_SIZE = 5000 

print(f"--- Bắt đầu phân tích SHAP cho {FOLD_NAME} ---")

# ----- 4. Tải Model -----
print(f"Đang tải model: {MODEL_FILE}")
try:
    model = CatBoostRegressor()
    model.load_model(MODEL_FILE)
except Exception as e:
    print(f"Lỗi tải model .cbm: {e}. Thử tải file .pkl...")
    MODEL_FILE_PKL = os.path.join("models_catboost", f"catboost_{FOLD_NAME}.pkl")
    model = joblib.load(MODEL_FILE_PKL)

# ----- 5. Tải và Chuẩn bị Dữ liệu Test -----
print(f"Đang tải dữ liệu test: {TEST_FILE}")
test_df = pd.read_csv(TEST_FILE, parse_dates=[DATE_COL])

# Quan trọng: Tái tạo lại feature_cols và cat_cols GIỐNG HỆT lúc train
print("Tái tạo feature_cols và cat_cols...")
if 'sales_true' in test_df.columns:
    # File test này có thể chứa các cột thừa (sales_true, sales_pred)
    # Chúng ta cần đảm bảo loại bỏ chúng
    cols_to_exclude = EXCLUDE_COLS + [
        'sales_true', 'sales_pred', 'fold'
    ]
    feature_cols = [c for c in test_df.columns if c not in cols_to_exclude]
else:
    feature_cols = [c for c in test_df.columns if c not in EXCLUDE_COLS]

cat_cols = [c for c in feature_cols if test_df[c].dtype == "object" or str(test_df[c].dtype).startswith("category")]

print(f"Tìm thấy {len(feature_cols)} features.")
print(f"Tìm thấy {len(cat_cols)} categorical features: {cat_cols[:5]}...")

# Áp dụng CÙNG MỘT bước tiền xử lý (fillna) như đã làm trong file train
print("Áp dụng fillna(0) cho features...")
test_df[feature_cols] = test_df[feature_cols].fillna(0)

# Tạo X_test (chỉ chứa các cột feature)
X_test = test_df[feature_cols]

# ----- 6. Lấy mẫu (Sampling) -----
if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(X_test):
    print(f"Lấy mẫu {SAMPLE_SIZE} dòng từ X_test để tính SHAP...")
    X_sample = X_test.sample(n=SAMPLE_SIZE, random_state=42)
else:
    print(f"Sử dụng toàn bộ {len(X_test)} dòng để tính SHAP (việc này có thể rất chậm)...")
    X_sample = X_test

# ----- 7. Tính toán SHAP -----
print("Khởi tạo SHAP TreeExplainer...")
# CatBoost thường hoạt động tốt nhất với TreeExplainer
explainer = shap.TreeExplainer(model)

print(f"Đang tính toán SHAP values cho {len(X_sample)} dòng. Xin chờ...")
# Đây là bước tốn thời gian nhất
shap_values = explainer.shap_values(X_sample)
print("Tính SHAP values hoàn tất.")

# ----- 8. Vẽ và Lưu Biểu đồ -----

# A. SUMMARY PLOT (DOT) - Biểu đồ quan trọng nhất
# Cho thấy feature quan trọng nhất VÀ hướng tác động
print("Đang vẽ Summary Plot (dot)...")
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
summary_dot_path = os.path.join(SHAP_OUTPUT_DIR, f"catboost_{FOLD_NAME}_summary_dot.png")
plt.savefig(summary_dot_path, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {summary_dot_path}")

# B. SUMMARY PLOT (BAR) - Biểu đồ xếp hạng feature
print("Đang vẽ Summary Plot (bar)...")
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
summary_bar_path = os.path.join(SHAP_OUTPUT_DIR, f"catboost_{FOLD_NAME}_summary_bar.png")
plt.savefig(summary_bar_path, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {summary_bar_path}")

# C. DEPENDENCE PLOTS - Biểu đồ phụ thuộc (Cho các feature quan trọng nhất)
# Lấy top 5 features từ summary bar
try:
    # Tính toán feature importance trung bình
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_sample.columns
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': mean_abs_shap})
    df_imp = df_imp.sort_values('importance', ascending=False)
    TOP_N_FEATURES = df_imp['feature'].head(5).tolist()
    print(f"Top 5 features: {TOP_N_FEATURES}")
except Exception:
    # Nếu lỗi, dùng các feature bạn đã đề cập
    TOP_N_FEATURES = ['lag_7', 'rolling_mean_30', 'onpromotion']

print("Đang vẽ Dependence Plots cho top features...")
for feature in TOP_N_FEATURES:
    if feature in X_sample.columns:
        print(f"  - Đang vẽ cho: {feature}")
        plt.figure()
        try:
            shap.dependence_plot(feature, shap_values, X_sample, show=False)
            dep_path = os.path.join(SHAP_OUTPUT_DIR, f"catboost_{FOLD_NAME}_dependence_{feature}.png")
            plt.savefig(dep_path, bbox_inches='tight')
            plt.close()
            print(f"    Đã lưu: {dep_path}")
        except Exception as e:
            print(f"    Lỗi khi vẽ dependence plot cho '{feature}': {e}")
            plt.close()

print("\n--- Hoàn tất! ---")
print(f"Tất cả hình ảnh SHAP đã được lưu vào thư mục: {SHAP_OUTPUT_DIR}")