"""
train_catboost_folds_with_r2_and_family_perf.py
Huấn luyện CatBoostRegressor theo 5 fold thời gian (thêm R^2 per-fold, overall và báo cáo family_performance).
Chú thích & hướng dẫn bằng tiếng Việt.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from catboost import CatBoostRegressor, Pool
import traceback, sys

# ----- Cấu hình -----
DATA_CSV = "final_dataset.csv"   # <-- chỉnh lại đường dẫn nếu cần
OUTPUT_MODELS_DIR = "models_catboost"
OUTPUT_RESULTS_DIR = "results_catboost"
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

# Nếu muốn lưu file train/test cho từng fold (train_fold_1.csv, test_fold_1.csv)
SAVE_FOLD_FILES = True

# CatBoost hyperparams (bạn có thể điều chỉnh)
cat_params = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "random_seed": 42,
    "thread_count": 8,
    "verbose": 100,
}

# Fold definitions (giống LightGBM)
folds = [
    ("fold_1", "2016-01-01", "2016-03-31", "2013-01-01", "2015-12-31"),
    ("fold_2", "2016-04-01", "2016-06-30", "2013-01-01", "2016-03-31"),
    ("fold_3", "2016-07-01", "2016-09-30", "2013-01-01", "2016-06-30"),
    ("fold_4", "2016-10-01", "2016-12-31", "2013-01-01", "2016-09-30"),
    ("fold_5", "2017-01-01", "2017-03-31", "2013-01-01", "2016-12-31"),
]

TARGET_COL = "sales"
DATE_COL = "date"
EXCLUDE_COLS = [TARGET_COL, DATE_COL]

# ----- Hàm metric -----
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    denom = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# ----- Load data -----
print("Đang load dữ liệu từ:", DATA_CSV)
df = pd.read_csv(DATA_CSV, parse_dates=[DATE_COL])
df = df[df[DATE_COL] >= pd.to_datetime("2013-01-01")].copy()
df.sort_values(DATE_COL, inplace=True)
df.reset_index(drop=True, inplace=True)
print("Dữ liệu n_rows:", len(df), "n_cols:", len(df.columns))

# Kiểm tra cột cần thiết
if TARGET_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột target '{TARGET_COL}' trong dữ liệu.")
if DATE_COL not in df.columns:
    raise ValueError(f"Không tìm thấy cột date '{DATE_COL}' trong dữ liệu.")

# ----- Chọn feature (loại date & target) -----
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print("Feature cols (ví dụ hiển thị 20 đầu):", feature_cols[:20])

# Convert các cột object sang category (CatBoost thích string/object hoặc category)
cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
print("Detected categorical columns:", cat_cols)

# Điền NA cho features (tùy bạn có thể thay bằng median/ffill)
df[feature_cols] = df[feature_cols].fillna(0)

# ----- Tiny debug train (kiểm tra môi trường CatBoost) -----
try:
    print("Thử tiny CatBoost fit (sample 5000) để kiểm tra):")
    n_small = min(5000, len(df))
    sample_idx = np.random.choice(len(df), n_small, replace=False)
    X_small = df.iloc[sample_idx][feature_cols]
    y_small = df.iloc[sample_idx][TARGET_COL].values
    pool_small = Pool(X_small, label=y_small, cat_features=cat_cols if len(cat_cols)>0 else None)
    tmp = CatBoostRegressor(**{**cat_params, "iterations":100, "thread_count":1, "verbose":0})
    tmp.fit(pool_small)
    print("Tiny CatBoost fit succeeded.")
except Exception:
    print("Tiny CatBoost fit failed; traceback:")
    traceback.print_exc(file=sys.stdout)

# ----- Chạy từng fold -----
metrics_list = []
all_preds = []

for (fold_name, test_start_s, test_end_s, train_start_s, train_end_s) in folds:
    print(f"\n--- Bắt đầu {fold_name} ---")
    test_start = pd.to_datetime(test_start_s)
    test_end = pd.to_datetime(test_end_s)
    train_start = pd.to_datetime(train_start_s)
    train_end = pd.to_datetime(train_end_s)

    train_df = df[(df[DATE_COL] >= train_start) & (df[DATE_COL] <= train_end)].copy()
    test_df  = df[(df[DATE_COL] >= test_start) & (df[DATE_COL] <= test_end)].copy()

    if train_df.empty or test_df.empty:
        print(f"Train hoặc test rỗng cho {fold_name} -> bỏ qua.")
        continue

    print(f"{fold_name} - n_train: {len(train_df)}, n_test: {len(test_df)}")

    # --- Lưu file train/test cho fold (nếu bật) ---
    if SAVE_FOLD_FILES:
        train_fp = os.path.join(OUTPUT_RESULTS_DIR, f"train_{fold_name}.csv")   # train_fold_1.csv
        test_fp = os.path.join(OUTPUT_RESULTS_DIR, f"test_{fold_name}.csv")     # test_fold_1.csv
        # Lưu nguyên dataframe (bao gồm tất cả cột). Nếu muốn lưu subset (date,store_nbr,family,sales) thay đổi ở đây.
        train_df.to_csv(train_fp, index=False)
        test_df.to_csv(test_fp, index=False)
        print(f"Saved fold files: {train_fp} ({len(train_df)} rows), {test_fp} ({len(test_df)} rows)")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    # Tách validation nhỏ (lấy 30 ngày cuối của train nếu đủ)
    val_size_days = 30
    val_cutoff = train_df[DATE_COL].max() - pd.Timedelta(days=val_size_days)
    val_df = train_df[train_df[DATE_COL] > val_cutoff]
    train_df_trim = train_df[train_df[DATE_COL] <= val_cutoff]

    if len(val_df) >= 50 and len(train_df_trim) > 0:
        X_tr = train_df_trim[feature_cols]
        y_tr = train_df_trim[TARGET_COL].values
        X_val = val_df[feature_cols]
        y_val = val_df[TARGET_COL].values
        use_val = True
        print("Sử dụng validation nội bộ:", len(X_tr), "train rows; val rows:", len(X_val))
    else:
        X_tr = X_train
        y_tr = y_train
        X_val = None
        y_val = None
        use_val = False
        print("Không tách validation nội bộ; dùng toàn bộ train để fit.")

    # Pool cho CatBoost: chấp nhận danh sách tên cột phân loại
    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_cols if len(cat_cols)>0 else None)
    if use_val:
        val_pool = Pool(X_val, label=y_val, cat_features=cat_cols if len(cat_cols)>0 else None)
    else:
        val_pool = None

    # Khởi tạo model CatBoost
    model = CatBoostRegressor(**cat_params)

    # Huấn luyện với early_stopping_rounds (CatBoost hỗ trợ tham số này)
    try:
        if use_val:
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, use_best_model=True)
        else:
            model.fit(train_pool)
    except Exception:
        print("Lỗi khi fit CatBoost; in traceback:")
        traceback.print_exc(file=sys.stdout)
        continue

    # Dự đoán trên test
    try:
        y_pred = model.predict(X_test)
    except Exception:
        test_pool = Pool(X_test, cat_features=cat_cols if len(cat_cols)>0 else None)
        y_pred = model.predict(test_pool)

    # Tính metrics (thêm R^2)
    fold_rmse = rmse(y_test, y_pred)
    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_mape = mape(y_test, y_pred)
    fold_r2 = r2_score(y_test, y_pred)
    print(f"{fold_name} - RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}, MAPE: {fold_mape:.2f}%, R2: {fold_r2:.4f}")

    # Lưu model: CatBoost có method save_model (native .cbm hoặc .json)
    model_fp = os.path.join(OUTPUT_MODELS_DIR, f"catboost_{fold_name}.cbm")
    try:
        model.save_model(model_fp)
        print("Saved CatBoost native model to:", model_fp)
    except Exception:
        joblib_fp = os.path.join(OUTPUT_MODELS_DIR, f"catboost_{fold_name}.pkl")
        joblib.dump(model, joblib_fp)
        print("Saved CatBoost model via joblib to:", joblib_fp)

    # Lưu dự đoán
    preds_df = test_df[[DATE_COL]].copy()
    if "store_nbr" in test_df.columns:
        preds_df["store_nbr"] = test_df["store_nbr"].values
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

# Lưu bảng metrics tổng hợp
metrics_df = pd.DataFrame(metrics_list)
metrics_fp = os.path.join(OUTPUT_RESULTS_DIR, "metrics_summary_catboost.csv")
metrics_df.to_csv(metrics_fp, index=False)
print("\nSaved metrics summary to:", metrics_fp)
print(metrics_df)

# Lưu tất cả dự đoán ghép lại và tính overall metrics (của all test rows)
family_perf_fp = os.path.join(OUTPUT_RESULTS_DIR, "family_performance_catboost.csv")
if len(all_preds) > 0:
    all_preds_df = pd.concat(all_preds, axis=0).reset_index(drop=True)
    all_preds_fp = os.path.join(OUTPUT_RESULTS_DIR, "all_folds_predictions_catboost.csv")
    all_preds_df.to_csv(all_preds_fp, index=False)
    print("Saved all folds predictions to:", all_preds_fp)

    # Compute overall metrics including overall R2
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
        perf_df = all_preds_df.dropna(subset=["sales_true","sales_pred","family"]).copy()

        def safe_r2(y_true, y_pred):
            try:
                return float(r2_score(y_true, y_pred))
            except Exception:
                return float("nan")

        rows = []
        grp = perf_df.groupby("family")
        for fam, g in grp:
            y_t = g["sales_true"].values
            y_p = g["sales_pred"].values
            mae_v = mean_absolute_error(y_t, y_p)
            rmse_v = rmse(y_t, y_p)
            mape_v = mape(y_t, y_p)
            r2_v = safe_r2(y_t, y_p)
            count_v = len(g)
            accuracy_v = 100.0 - mape_v
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
        family_perf = family_perf.sort_values("R2", ascending=False).reset_index(drop=True)

        # làm tròn để dễ đọc
        for c in ["MAE","RMSE","MAPE","R2","Accuracy (%)"]:
            if c in family_perf.columns:
                family_perf[c] = family_perf[c].round(6)

        family_perf.to_csv(family_perf_fp, index=False)
        print("Saved family performance to:", family_perf_fp)
        print(family_perf.head(20))
    else:
        print("Không tìm thấy cột 'family' trong all_preds_df — không thể tạo báo cáo family_performance.")

    # Thêm 1 dòng tổng hợp vào metrics_df
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
    metrics_df.to_csv(metrics_fp, index=False)
    print("\nUpdated metrics summary (with overall row) saved to:", metrics_fp)
    print(metrics_df)

print("\nHoàn tất huấn luyện 5 fold CatBoost.")
