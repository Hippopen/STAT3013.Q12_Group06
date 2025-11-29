"""
train_lightgbm_folds_fixed_with_r2_and_family_perf.py
Phiên bản: giống file gốc + sau khi huấn luyện sẽ sinh file CSV "family_performance.csv"
Chú thích bằng tiếng Việt.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from lightgbm import LGBMRegressor
import lightgbm as lgb

# ----- Cấu hình -----
DATA_CSV = "final_dataset.csv"   # <-- chỉnh lại đường dẫn nếu cần
OUTPUT_MODELS_DIR = "models"
OUTPUT_RESULTS_DIR = "results"
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

# Nếu muốn lưu file train/test cho từng fold (train_fold_1.csv, test_fold_1.csv)
SAVE_FOLD_FILES = True

# LightGBM hyperparams (bạn có thể chỉnh)
lgb_params = {
    "objective": "regression",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_jobs": -1
}

# Các fold test theo yêu cầu (test gồm 3 tháng liên tiếp)
folds = [
    ("fold_1", "2016-01-01", "2016-03-31", "2013-01-01", "2015-12-31"),
    ("fold_2", "2016-04-01", "2016-06-30", "2013-01-01", "2016-03-31"),
    ("fold_3", "2016-07-01", "2016-09-30", "2013-01-01", "2016-06-30"),
    ("fold_4", "2016-10-01", "2016-12-31", "2013-01-01", "2016-09-30"),
    ("fold_5", "2017-01-01", "2017-03-31", "2013-01-01", "2016-12-31"),
]

# Các cột cần loại bỏ/không dùng làm feature
TARGET_COL = "sales"
DATE_COL = "date"
EXCLUDE_COLS = [TARGET_COL, DATE_COL]  # những cột không dùng làm X

# ----- Hàm đánh giá -----
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    # Tránh chia cho 0 bằng cách thay mẫu 0 bằng 1 trong mẫu chia (tùy chọn)
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# ----- Load data -----
print("Đang load dữ liệu từ:", DATA_CSV)
df = pd.read_csv(DATA_CSV, parse_dates=[DATE_COL])
# Lọc dữ liệu tối thiểu bắt đầu từ 2013-01-01
df = df[df[DATE_COL] >= pd.to_datetime("2013-01-01")].copy()
df.sort_values(DATE_COL, inplace=True)
df.reset_index(drop=True, inplace=True)
print("Dữ liệu n_rows:", len(df), "n_cols:", len(df.columns))

# ----- Kiểm tra cột cần thiết -----
if TARGET_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột target '{TARGET_COL}' trong dữ liệu.")
if DATE_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột date '{DATE_COL}' trong dữ liệu.")

# ----- Chọn features tự động (loại bỏ date và target) -----
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print("Sử dụng features (ví dụ, hiển thị 20 đầu):", feature_cols[:20])

# Chuyển các cột object sang category (an toàn và chặt chẽ)
from pandas.api.types import is_object_dtype, is_categorical_dtype, is_numeric_dtype

# Xác định cột categorical: nếu dtype là object hoặc đã là categorical
cat_feature_cols = [c for c in feature_cols if is_object_dtype(df[c]) or is_categorical_dtype(df[c])]
for c in cat_feature_cols:
    # Điền NaN bằng chuỗi 'missing' trước khi convert để tránh lỗi thêm category sau
    df[c] = df[c].fillna('missing')
    # Ép về Pandas Categorical (an toàn với cả object/mixed)
    df[c] = pd.Categorical(df[c])
print("Cột categorical sau khi chuẩn hoá:", cat_feature_cols)

# Các cột còn lại coi là numeric (hoặc bool); nếu không numeric thì ta vẫn fillna bằng 0
num_feature_cols = [c for c in feature_cols if c not in cat_feature_cols]

# Điền NaN cho numeric features bằng 0 (hoặc bạn có thể đổi sang median/groupby)
if len(num_feature_cols) > 0:
    df[num_feature_cols] = df[num_feature_cols].fillna(0)

# Sau khi chuẩn hoá, đảm bảo không còn dtype object trong feature set
bad_obj = [c for c in feature_cols if df[c].dtype == 'object']
if len(bad_obj) > 0:
    print("Cảnh báo: còn cột object chưa được convert:", bad_obj)
    # Thử ép kiểu an toàn lại
    for c in bad_obj:
        df[c] = pd.Categorical(df[c].fillna('missing'))
    print("Đã ép lại các cột này về category.")

# ----- Chạy từng fold -----
metrics_list = []
all_preds = []
for i, (fold_name, test_start_s, test_end_s, train_start_s, train_end_s) in enumerate(folds, start=1):
    print(f"\n--- Bắt đầu {fold_name} ---")
    test_start = pd.to_datetime(test_start_s)
    test_end = pd.to_datetime(test_end_s)
    train_start = pd.to_datetime(train_start_s)
    train_end = pd.to_datetime(train_end_s)

    # Lấy train và test theo thời gian
    train_df = df[(df[DATE_COL] >= train_start) & (df[DATE_COL] <= train_end)].copy()
    test_df = df[(df[DATE_COL] >= test_start) & (df[DATE_COL] <= test_end)].copy()

    if train_df.empty or test_df.empty:
        print(f"Lỗi: train hoặc test rỗng cho {fold_name}. Train rows={len(train_df)}, Test rows={len(test_df)}")
        continue

    print(f"{fold_name} - Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # --- Lưu file train/test cho fold (nếu bật) ---
    if SAVE_FOLD_FILES:
        train_fp = os.path.join(OUTPUT_RESULTS_DIR, f"train_{fold_name}.csv")   # -> train_fold_1.csv
        test_fp = os.path.join(OUTPUT_RESULTS_DIR, f"test_{fold_name}.csv")     # -> test_fold_1.csv
        # Lưu nguyên dataframe (bao gồm tất cả cột) - bạn có thể chọn lưu subset nếu muốn
        train_df.to_csv(train_fp, index=False)
        test_df.to_csv(test_fp, index=False)
        print(f"Saved fold files: {train_fp} ({len(train_df)} rows), {test_fp} ({len(test_df)} rows)")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    # ----- Tạo validation nhỏ từ cuối train để early stopping (tùy chọn) -----
    val_size_days = 30
    val_cutoff = train_df[DATE_COL].max() - pd.Timedelta(days=val_size_days)
    val_df = train_df[train_df[DATE_COL] > val_cutoff]
    train_df_trim = train_df[train_df[DATE_COL] <= val_cutoff]

    use_val = False
    if len(val_df) >= 50 and len(train_df_trim) > 0:
        use_val = True
        X_tr = train_df_trim[feature_cols]
        y_tr = train_df_trim[TARGET_COL].values
        X_val = val_df[feature_cols]
        y_val = val_df[TARGET_COL].values
        print(f"Sử dụng validation nội bộ: train_trim_rows={len(X_tr)}, val_rows={len(X_val)}")
    else:
        X_tr = X_train
        y_tr = y_train
        X_val = None
        y_val = None
        print("Không có validation nội bộ (dùng toàn bộ train để huấn luyện).")

    # ----- Khởi tạo model và huấn luyện -----
    model = LGBMRegressor(**lgb_params)
    fit_kwargs = {}
    if use_val and X_val is not None:
        fit_kwargs["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]
        fit_kwargs["eval_metric"] = "rmse"
        # Dùng callbacks để early stopping (tương thích với các phiên bản lightgbm)
        fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]

    # truyền danh sách cột categorical (tên cột) vào fit
    categorical_feature = [c for c in feature_cols if str(X_tr[c].dtype) == "category"]
    if len(categorical_feature) > 0:
        fit_kwargs["categorical_feature"] = categorical_feature

    print("Bắt đầu huấn luyện model LightGBM...")
    model.fit(X_tr, y_tr, **fit_kwargs)
    print("Huấn luyện xong. Best iteration (n_estimators):", getattr(model, "best_iteration_", None) or lgb_params["n_estimators"])

    # ----- Dự đoán trên test và đánh giá -----
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_ if getattr(model, "best_iteration_", None) else None)
    fold_rmse = rmse(y_test, y_pred)
    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_mape = mape(y_test, y_pred)
    fold_r2 = r2_score(y_test, y_pred)

    print(f"{fold_name} - RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}, MAPE: {fold_mape:.2f}%, R2: {fold_r2:.4f}")

    # ----- Lưu model và kết quả -----
    model_fp = os.path.join(OUTPUT_MODELS_DIR, f"lightgbm_{fold_name}.pkl")
    joblib.dump(model, model_fp)
    print("Saved model to:", model_fp)

    # Lưu dự đoán test (kèm cột date, store_nbr,... nếu có)
    preds_df = test_df[[DATE_COL]].copy()
    if "store_nbr" in test_df.columns:
        preds_df["store_nbr"] = test_df["store_nbr"].values
    # lưu tên family dưới cột 'family' (giữ tương thích với pipeline hiện tại)
    if "family" in test_df.columns:
        preds_df["family"] = test_df["family"].values
    preds_df["sales_true"] = y_test
    preds_df["sales_pred"] = y_pred
    preds_fp = os.path.join(OUTPUT_RESULTS_DIR, f"predictions_{fold_name}.csv")
    preds_df.to_csv(preds_fp, index=False)
    print("Saved predictions to:", preds_fp)

    metrics_list.append({
        "fold": fold_name,
        "train_start": train_start_s,
        "train_end": train_end_s,
        "test_start": test_start_s,
        "test_end": test_end_s,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "rmse": fold_rmse,
        "mae": fold_mae,
        "mape": fold_mape,
        "r2": fold_r2
    })

    preds_df["fold"] = fold_name
    all_preds.append(preds_df)

# ----- Lưu bảng metrics tổng hợp -----
metrics_df = pd.DataFrame(metrics_list)
metrics_fp = os.path.join(OUTPUT_RESULTS_DIR, "metrics_summary_lightgbm.csv")
metrics_df.to_csv(metrics_fp, index=False)
print("\nSaved metrics summary to:", metrics_fp)
print(metrics_df)

# ----- (Tùy chọn) Lưu toàn bộ dự đoán ghép lại và tính metrics tổng hợp -----
family_perf_fp = os.path.join(OUTPUT_RESULTS_DIR, "family_performance.csv")
if len(all_preds) > 0:
    all_preds_df = pd.concat(all_preds, axis=0).reset_index(drop=True)
    all_preds_fp = os.path.join(OUTPUT_RESULTS_DIR, "all_folds_predictions_lightgbm.csv")
    all_preds_df.to_csv(all_preds_fp, index=False)
    print("Saved all folds predictions to:", all_preds_fp)

    # Tính metrics chung trên toàn bộ test rows
    y_all_true = all_preds_df["sales_true"].values
    y_all_pred = all_preds_df["sales_pred"].values
    overall_rmse = rmse(y_all_true, y_all_pred)
    overall_mae = mean_absolute_error(y_all_true, y_all_pred)
    overall_mape = mape(y_all_true, y_all_pred)
    overall_r2 = r2_score(y_all_true, y_all_pred)

    print("\n----- Overall (tổng hợp all folds) -----")
    print(f"Total test rows: {len(all_preds_df)}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall MAPE: {overall_mape:.2f}%")
    print(f"Overall R2: {overall_r2:.4f}")

    # ----- TẠO BÁO CÁO family_performance (theo cột 'family') -----
    if "family" in all_preds_df.columns:
        print("Tính family_performance từ all folds predictions...")
        # loại bỏ dòng thiếu giá trị
        perf_df = all_preds_df.dropna(subset=["sales_true","sales_pred","family"]).copy()

        # hàm safe r2 để tránh lỗi khi biến độc lập
        def safe_r2(y_true, y_pred):
            try:
                return float(r2_score(y_true, y_pred))
            except Exception:
                return float("nan")

        # groupby family tính các chỉ số
        grp = perf_df.groupby("family")
        rows = []
        for fam, g in grp:
            y_t = g["sales_true"].values
            y_p = g["sales_pred"].values
            mae_v = mean_absolute_error(y_t, y_p)
            rmse_v = rmse(y_t, y_p)
            mape_v = mape(y_t, y_p)
            r2_v = safe_r2(y_t, y_p)
            count_v = len(g)
            accuracy_v = 100.0 - mape_v  # cùng chuẩn với file mẫu bạn gửi
            rows.append({
                "family_name": fam,
                "MAE": mae_v,
                "RMSE": rmse_v,
                "MAPE": mape_v,
                "R2": r2_v,
                "Accuracy (%)": accuracy_v,
                "Count": count_v
            })

        family_perf = pd.DataFrame(rows)
        # sắp xếp descending theo R2 giống file mẫu (tuỳ bạn)
        family_perf = family_perf.sort_values("R2", ascending=False).reset_index(drop=True)

        # Làm tròn các cột số cho dễ nhìn
        for c in ["MAE","RMSE","MAPE","R2","Accuracy (%)"]:
            if c in family_perf.columns:
                family_perf[c] = family_perf[c].round(6)  # giữ đủ chữ số; bạn có thể đổi round(3)

        # Lưu CSV
        family_perf.to_csv(family_perf_fp, index=False)
        print("Saved family performance to:", family_perf_fp)
        print(family_perf.head(20))
    else:
        print("Không tìm thấy cột 'family' trong all_preds_df — không thể tạo báo cáo family_performance.")

    # Thêm 1 dòng tổng hợp vào metrics_df (nếu muốn lưu chung)
    summary_row = {
        "fold": "all_folds",
        "train_start": "",
        "train_end": "",
        "test_start": "",
        "test_end": "",
        "n_train_rows": "",
        "n_test_rows": len(all_preds_df),
        "rmse": overall_rmse,
        "mae": overall_mae,
        "mape": overall_mape,
        "r2": overall_r2
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([summary_row])], axis=0, ignore_index=True)
    # Lưu lại metrics có dòng tổng hợp
    metrics_df.to_csv(metrics_fp, index=False)
    print("\nUpdated metrics summary (with overall row) saved to:", metrics_fp)
    print(metrics_df)

print("\nHoàn tất huấn luyện 5 fold LightGBM.")