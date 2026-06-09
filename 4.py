import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import white_tophat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================================================
# 1. 自定义设置你的 Excel 文件存放位置 (支持绝对路径)
# =========================================================================
DATA_FOLDER = r"C:/Users/Surface pro8/Desktop/多晶"

file_list = [
    "jh.xlsx",
    "wzh.xlsx",
    "ljl.xlsx",
    "syf.xlsx",
    "xdh.xlsx",
    "ypg.xlsx",
    "zbs.xlsx",
]

dfs = {}
for f in file_list:
    full_path = os.path.join(DATA_FOLDER, f)
    if not os.path.exists(full_path):
        csv_backup_name = f.replace(".xlsx", "") + " - Sheet1.csv"
        csv_full_path = os.path.join(DATA_FOLDER, csv_backup_name)
        if os.path.exists(csv_full_path):
            df_temp = pd.read_csv(
                csv_full_path, header=None, skiprows=1, names=["x", "y"]
            )
        else:
            raise FileNotFoundError(f"找不到文件: {f}")
    else:
        df_temp = pd.read_excel(full_path)
        df_temp.columns = ["x", "y"]
    dfs[f] = df_temp

# =========================================================================
# 2. 数据对齐与核心预处理：剥离缓变背景 (FILTER_SIZE=20)
# =========================================================================
base_x = dfs[file_list[0]]["x"].values
combined_df = pd.DataFrame({"x": base_x})

# FILTER_SIZE 决定了多宽的鼓包被当作背景滤除。20 左右能完美剥离大坡，只保留尖尖
FILTER_SIZE = 20 

for f in file_list:
    name = f.split(".")[0]
    y_raw = dfs[f]["y"].values
    # 在对齐前先用顶帽变换扣除缓变背景
    y_pure_peaks = white_tophat(y_raw, size=FILTER_SIZE)
    combined_df[name] = y_pure_peaks

cleaned_df = combined_df.dropna().reset_index(drop=True)
x_clean = cleaned_df["x"].values
Y_signals = cleaned_df.drop(columns=["x"]).values

# =========================================================================
# 3. 对去除背景后的纯尖锐峰进行标准化与 PCA 提取
# =========================================================================
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y_signals)

pca = PCA(n_components=1)
Y_pca = pca.fit_transform(Y_scaled)
Y_reconstructed = pca.inverse_transform(Y_pca)
y_pca1 = np.mean(Y_reconstructed, axis=1)

# =========================================================================
# 4. 指定你要标注的 17 个精确峰横坐标数据
# =========================================================================
image_peak_x = [6.2495, 7.207, 10.24535, 12.03453, 12.5828, 14.56819, 15.907, 
                16.335, 17.939, 19.07012, 20.84, 21.84, 22.16, 23.437, 24.35, 
                24.65, 25.826]

# =========================================================================
# 5. 可视化绘图展示（单大图、去背景PCA1主线、多色递减竖虚线、无小框）
# =========================================================================
plt.figure(figsize=(12, 6), dpi=120)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 绘制已经完美“剔除缓变大鼓包”的纯净 PCA1 尖锐峰信号
plt.plot(x_clean, y_pca1, color="black", linewidth=2.5, label="纯尖锐峰 PCA1 主成分")

# -------------------------------------------------------------------------
# 核心视觉映射：按明暗显著程度依次递减的原则配置虚线
# -------------------------------------------------------------------------
for idx, px in enumerate(image_peak_x):
    peak_number = idx + 1  # 编号从 1 开始

    if peak_number in [2, 3, 8]:
        # 层级一：最鲜艳突出的深红色，线最粗
        line_color = "crimson"
        line_width = 1.2
    elif peak_number in [1, 5, 6, 9]:
        # 层级二：次显著的皇家蓝，线稍细
        line_color = "royalblue"
        line_width = 1.0
    else:
        # 层级三：视觉最弱、最内敛的淡灰色，线最细
        line_color = "lightgray"
        line_width = 0.8

    # 在无背景拉扯的干净尖峰位置画上漂亮的垂直虚线
    plt.axvline(
        x=px, color=line_color, linewidth=line_width, linestyle="--", alpha=0.9
    )

    # 在图表正上方淡淡标记出编号，方便肉眼快速对应查找
    plt.text(
        px,
        max(y_pca1) * 1.01,
        f"#{peak_number}",
        color=line_color,
        fontsize=8,
        ha="center",
        alpha=0.75,
    )

# 基础图表清洗，没有网格，没有多余小框，极其干净
plt.title("纯净 PCA1 信号多层级核心峰位置对应图 (已扣除缓变背景)", fontsize=12, pad=15)
plt.xlabel("角度", fontsize=10)
plt.ylabel("标准幅值 (已去背景)", fontsize=10)
plt.grid(False)  
plt.legend(loc="upper right", fontsize=10, frameon=True)
plt.tight_layout()
plt.show()