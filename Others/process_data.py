import pandas as pd
import os

# 1. Danh sách file và Model tương ứng
file_map = {
    'catboost_family_performance_quantile.csv': 'CatBoost',
    'lgbm_family_performance_quantile.csv': 'LightGBM',
    'linear_family_performance_quantile.csv': 'Linear',
    'LSTM_family_performance_quantile.csv': 'LSTM',
    'PROPHET_family_perfomance_quantile.csv': 'Prophet',
    'rf_family_performance_quantile.csv': 'RandomForest',
    'xgb_family_performance_quantile.csv': 'XGBoost'
}

dfs = []
print("Đang đọc dữ liệu...")

# 2. Đọc dữ liệu
for fname, model_name in file_map.items():
    if os.path.exists(fname):
        try:
            df = pd.read_csv(fname)
            df['model'] = model_name
            if 'mean_sales' in df.columns:
                df.rename(columns={'mean_sales': 'mean_sale'}, inplace=True)
            dfs.append(df)
        except Exception as e:
            pass

if dfs:
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Các chỉ số cần so sánh
    metrics = ['WAPE', 'R2', 'MAE', 'RMSE', 'MAPE', 'accuracy', 'coverage_95', 'avg_width', 'mean_sale']
    available_metrics = [m for m in metrics if m in full_df.columns]

    # --- 1. Tạo bảng OVERALL (So sánh tổng thể) ---
    # Gom nhóm theo Model (bỏ qua Segment) và tính trung bình
    overall_df = full_df.groupby(['model'])[available_metrics].mean().reset_index()
    if 'WAPE' in overall_df.columns:
        overall_df.sort_values('WAPE', ascending=True, inplace=True)
    
    overall_df.to_csv('summary_overall.csv', index=False)
    print("-> Đã tạo file: summary_overall.csv")

    # --- 2. Tạo bảng theo từng SEGMENT (High, Medium, Low) ---
    target_segments = ['High', 'Medium', 'Low']
    
    # Tính trung bình theo Segment + Model trước
    segment_summary_df = full_df.groupby(['segment', 'model'])[available_metrics].mean().reset_index()

    for seg in target_segments:
        # Lọc dữ liệu của segment
        seg_df = segment_summary_df[segment_summary_df['segment'] == seg].copy()
        
        if not seg_df.empty:
            # Sắp xếp theo WAPE tốt nhất
            if 'WAPE' in seg_df.columns:
                seg_df.sort_values('WAPE', ascending=True, inplace=True)
            
            fname = f'summary_{seg}.csv'
            seg_df.to_csv(fname, index=False)
            print(f"-> Đã tạo file: {fname}")
        else:
            print(f" [!] Không có dữ liệu cho segment: {seg}")

    print("\nHoàn tất! Kiểm tra thư mục để lấy 4 file.")
else:
    print("Không tìm thấy dữ liệu đầu vào.")