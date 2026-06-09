import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================================================
# 1. 自定义设置你的 Excel 文件存放位置 (支持绝对路径)
# =========================================================================
DATA_FOLDER = r"C:/Users/Surface pro8/Desktop/多晶"

# 7 个需要融合的 Excel 文件名列表
file_list = [
    "jh.xlsx",
    "wzh.xlsx",
    "ljl.xlsx",
    "syf.xlsx",
    "xdh.xlsx",
    "ypg.xlsx",
    "zbs.xlsx",
]

print(f">>> 开始从目标路径加载本地数据: {DATA_FOLDER}")
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
            raise FileNotFoundError(
                f"\n【文件缺失错误】:\n在路径 '{DATA_FOLDER}' 下既找不到 '{f}' \n也找不到 '{csv_backup_name}'！\n请确认该文件夹下是否有这两个文件。"
            )
    else:
        df_temp = pd.read_excel(full_path)
        df_temp.columns = ["x", "y"]

    dfs[f] = df_temp

# =========================================================================
# 2. 多通道数据合并与缺失值清洗 (Alignment)
# =========================================================================
base_x = dfs[file_list[0]]["x"].values
combined_df = pd.DataFrame({"x": base_x})

for f in file_list:
    name = f.split(".")[0]
    combined_df[name] = dfs[f]["y"]

cleaned_df = combined_df.dropna().reset_index(drop=True)
x_clean = cleaned_df["x"].values
Y_signals = cleaned_df.drop(columns=["x"]).values

print(
    f">>> 数据对齐与清洗完毕。参与PCA融合的数据点共 {len(x_clean)} 个，通道数：{Y_signals.shape[1]} 个。"
)

# =========================================================================
# 3. 幅度归一化 (Standardization)
# =========================================================================
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y_signals)

# =========================================================================
# 4. 主成分分析与信号重构 (PCA Denoising)
# =========================================================================
pca = PCA(n_components=7)
Y_pca = pca.fit_transform(Y_scaled)

# 第一主成分：只拿 PC1 进行逆变换重构
Y_pca1_only = np.zeros_like(Y_pca)
Y_pca1_only[:, 0] = Y_pca[:, 0]
Y_reconstructed_scaled_pc1 = pca.inverse_transform(Y_pca1_only)
y_final_standard_pc1 = np.mean(Y_reconstructed_scaled_pc1, axis=1)

# 第二主成分：只拿 PC2 进行逆变换重构
Y_pca2_only = np.zeros_like(Y_pca)
Y_pca2_only[:, 1] = Y_pca[:, 1]
Y_reconstructed_scaled_pc2 = pca.inverse_transform(Y_pca2_only)
y_final_standard_pc2 = np.mean(Y_reconstructed_scaled_pc2, axis=1)

print(
    f">>> 第一主成分(PC1)的方差贡献率: {pca.explained_variance_ratio_[0]*100:.2f}%"
)
print(
    f">>> 第二主成分(PC2)的方差贡献率: {pca.explained_variance_ratio_[1]*100:.2f}%"
)
print(
    f">>> 第三主成分(PC3)的方差贡献率: {pca.explained_variance_ratio_[2]*100:.2f}%"
)
print(
    f">>> 第四主成分(PC4)的方差贡献率: {pca.explained_variance_ratio_[3]*100:.2f}%"
)
print(
    f">>> 第五主成分(PC5)的方差贡献率: {pca.explained_variance_ratio_[4]*100:.2f}%"
)
print(
    f">>> 第六主成分(PC6)的方差贡献率: {pca.explained_variance_ratio_[5]*100:.2f}%"
)
print(
    f">>> 第七主成分(PC7)的方差贡献率: {pca.explained_variance_ratio_[6]*100:.2f}%"
)

# =========================================================================
# 5. 可视化绘图 (修改点：分为上下两个独立的子图)
# =========================================================================
# 适当增加了画布高度(从7改成9)，防止两个子图的文字挤在一起
plt.figure(figsize=(12, 9), dpi=120)

# 防止中文标题乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------------------------
# 图 1：PCA1 (第一主成分) 独立子图
# -------------------------------------------------------------------------
plt.subplot(2, 1, 1)

# 绘制原始背景曲线
for i, f in enumerate(file_list):
    name = f.split(".")[0]
    plt.plot(x_clean, Y_scaled[:, i], alpha=0.3, label=f"{name}")

# 绘制 PCA 第一主成分 (红线)
plt.plot(
    x_clean,
    y_final_standard_pc1,
    color="red",
    linewidth=3,
    label="PCA1 (第一主成分)",
)

plt.title("各通道信号与 PCA1 (第一主成分) 波形对比", fontsize=12)
plt.ylabel("标准幅值", fontsize=10)
plt.xlabel("角度", fontsize=10)
plt.grid(False)  # 去掉灰色虚线背景
plt.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=8)

# -------------------------------------------------------------------------
# 图 2：PCA2 (第二主成分) 独立子图
# -------------------------------------------------------------------------
plt.subplot(2, 1, 2)

# 绘制原始背景曲线
for i, f in enumerate(file_list):
    name = f.split(".")[0]
    plt.plot(x_clean, Y_scaled[:, i], alpha=0.3, label=f"{name}")

# 绘制 PCA 第二主成分 (蓝线)
plt.plot(
    x_clean,
    y_final_standard_pc2,
    color="blue",
    linewidth=3,
    label="PCA2 (第二主成分)",
)

plt.title("各通道信号与 PCA2 (第二主成分) 波形对比", fontsize=12)
plt.ylabel("标准幅值", fontsize=10)
plt.xlabel("角度", fontsize=10)
plt.grid(False)  # 去掉灰色虚线背景
plt.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=8)

# 自动调整整体间距并显示
plt.tight_layout()
plt.show()