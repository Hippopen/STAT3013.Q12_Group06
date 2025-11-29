import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIG & LOAD DATA
# ==============================================================================
print(">>> [1/5] Loading Data & Preparing...")

def find_file_path(filename, search_path='.'):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

SEGMENT_PATH = find_file_path('family_segments.csv')
TRAIN_FOLD_PATH = find_file_path('train_fold_1.csv')
TEST_FOLD_PATH = find_file_path('test_fold_1.csv')

if not SEGMENT_PATH or not TRAIN_FOLD_PATH or not TEST_FOLD_PATH:
    print("❌ LỖI: Không tìm thấy file dữ liệu.")
    exit()

segments = pd.read_csv(SEGMENT_PATH, sep=';')
medium_families = segments[segments['segment'] == 'Medium']['family_name'].unique()

train_df = pd.read_csv(TRAIN_FOLD_PATH)
test_df = pd.read_csv(TEST_FOLD_PATH)

train_m = train_df[train_df['family'].isin(medium_families)].copy()
test_m = test_df[test_df['family'].isin(medium_families)].copy()

# Preprocessing
train_m = train_m.fillna(0)
test_m = test_m.fillna(0)
train_m['date'] = pd.to_datetime(train_m['date'])
test_m['date'] = pd.to_datetime(test_m['date'])

# Clipping Outliers (để model không vẽ vọt lên trời)
for fam in medium_families:
    mask = train_m['family'] == fam
    threshold = train_m.loc[mask, 'sales'].quantile(0.99)
    threshold = max(threshold, 10) 
    train_m.loc[mask & (train_m['sales'] > threshold), 'sales'] = threshold

# Feature Engineering
cat_cols = ['family', 'city', 'state', 'type']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train_m[col], test_m[col]]).unique()
    le.fit(all_values)
    train_m[col] = le.transform(train_m[col])
    test_m[col] = le.transform(test_m[col])
    encoders[col] = le

features = [
    'store_nbr', 'family', 'onpromotion', 'transactions', 
    'city', 'state', 'type', 'cluster', 'dcoilwtico', 
    'is_holiday', 'day_of_week', 'month', 'year', 'is_weekend',
    'sales_lag_7', 'sales_lag_14', 'rolling_mean_30'
]

X_train = train_m[features]
y_train = train_m['sales']
X_test = test_m[features]
# weights = train_m['year'].apply(lambda x: 1.5 if x == 2015 else 1.0)

# ==============================================================================
# 2. TRAINING & PREDICT
# ==============================================================================
print(">>> [2/5] Retraining XGBoost Model...")

model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=1000, # Giảm nhẹ số cây để vẽ nhanh hơn (vẫn đủ chính xác để vẽ)
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    eval_metric='mae'
)

model.fit(X_train, y_train, verbose=False) # Tắt log

print(">>> [3/5] Predicting...")
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

test_m['predicted_sales'] = y_pred
# Giải mã tên family để lọc
test_m['family_name'] = encoders['family'].inverse_transform(test_m['family'])

# ==============================================================================
# 3. PLOTTING (VẼ BIỂU ĐỒ)
# ==============================================================================
print(">>> [4/5] Generating Plots...")

# Chọn 2 family điển hình để vẽ
target_families = ['LIQUOR,WINE,BEER', 'PREPARED FOODS']

# Thiết lập style cho đẹp
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10)) # Kích thước ảnh lớn

for i, fam in enumerate(target_families):
    if fam not in medium_families:
        continue
        
    # Lọc dữ liệu của family này
    df_fam = test_m[test_m['family_name'] == fam].copy()
    
    # Group by Date để tính tổng doanh số theo ngày (của tất cả cửa hàng cộng lại)
    # Điều này giúp biểu đồ mượt và dễ nhìn hơn là vẽ hàng nghìn store
    daily_sales = df_fam.groupby('date')[['sales', 'predicted_sales']].sum().reset_index()
    
    # Vẽ subplot
    plt.subplot(2, 1, i+1) # 2 hàng, 1 cột, hình thứ i+1
    
    # Vẽ đường Thực tế
    plt.plot(daily_sales['date'], daily_sales['sales'], 
             label='Actual Sales', color='#1f77b4', linewidth=2, alpha=0.8)
    
    # Vẽ đường Dự báo
    plt.plot(daily_sales['date'], daily_sales['predicted_sales'], 
             label='Predicted Sales (XGBoost)', color='#ff7f0e', linewidth=2, linestyle='--')
    
    plt.title(f'Forecast vs Actual: {fam} (Daily Aggregated)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Sales', fontsize=12)
    plt.xlabel('Date (2016)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'forecast_vs_actual_M.png'
plt.savefig(save_path, dpi=300) # Xuất ảnh nét (300 dpi)
plt.close()

print(f"\n✅ Đã xuất biểu đồ thành công: {save_path}")
print("   (Bạn có thể mở file ảnh này lên để đưa vào báo cáo)")