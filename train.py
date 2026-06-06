import numpy as np
import pandas as pd  # 新增 pandas 套件來處理 CSV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 步驟 1: 讀取並準備 CSV 訓練資料集
# ==========================================
csv_filename = "rolypoly_dataset.csv"
print(f"📂 正在讀取資料集: {csv_filename}...")

try:
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"❌ 找不到 {csv_filename}！請確認檔案跟這支程式在同一個資料夾。")
    exit()

# 定義我們要餵給 AI 的「特徵欄位」(X)
feature_cols = ["Max_Amplitude", "Mean_Period", "Period_CV", "Decay_Slope", "Mean_Risk"]

# 檢查 CSV 是否包含我們需要的欄位
missing_cols = [col for col in feature_cols + ["True_Weight_g"] if col not in df.columns]
if missing_cols:
    print(f"❌ CSV 檔案缺少必要的欄位：{missing_cols}")
    exit()

# 萃取特徵 (X) 與 實際重量標籤 (Y)
X_data = df[feature_cols].values
Y_labels = df["True_Weight_g"].values

print(f"✅ 成功載入 {len(df)} 筆不倒翁資料！\n")

# ==========================================
# 步驟 2: 訓練 AI 模型 (隨機森林迴歸器)
# ==========================================
print("🚀 開始訓練不倒翁重量預測模型...")

# 建立模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 讓模型學習：特徵(X) 與 重量(Y) 之間的關係
model.fit(X_data, Y_labels)

# 簡單評估一下模型在訓練集上的表現
y_pred_train = model.predict(X_data)
mae = mean_absolute_error(Y_labels, y_pred_train)
r2 = r2_score(Y_labels, y_pred_train)

print("✅ 模型訓練完成！")
print(f"📊 模型評估 -> 平均誤差: {mae:.2f} 克 | 準確度(R²): {r2:.4f}")

# ==========================================
# 步驟 3: 預測全新影片 (實戰應用範例)
# ==========================================
# 假設今天有一部全新的影片，OpenCV 跑完後得出以下特徵：
# [Max_Amplitude, Mean_Period, Period_CV, Decay_Slope, Mean_Risk]
new_video_features = np.array([[28.5, 0.715, 0.015, -2.55, 12.0]])

# 請 AI 預測重量
predicted_weight = model.predict(new_video_features)

print(f"\n🔮 [預測結果]")
print(f"根據輸入的物理特徵，系統預測此不倒翁重量為：{predicted_weight[0]:.1f} 克")