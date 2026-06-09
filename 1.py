import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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
# 2. 数据对齐与归一化
# =========================================================================
base_x = dfs[file_list[0]]["x"].values
combined_df = pd.DataFrame({"x": base_x})
for f in file_list:
    name = f.split(".")[0]
    combined_df[name] = dfs[f]["y"]

cleaned_df = combined_df.dropna().reset_index(drop=True)
x_clean = cleaned_df["x"].values
Y_signals = cleaned_df.drop(columns=["x"]).values

scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y_signals)

# =========================================================================
# 3. PCA 提取第一主成分波形 (PCA1)
# =========================================================================
pca = PCA(n_components=1)
Y_pca = pca.fit_transform(Y_scaled)
Y_reconstructed = pca.inverse_transform(Y_pca)
y_pca1 = np.mean(Y_reconstructed, axis=1)

# =========================================================================
# 4. 定义单峰高斯模型
# =========================================================================
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# 设定的三个拟合中心目标点
target_peaks = [7.2, 10.2, 16.3]
half_widths = [0.5, 0.5, 0.2]  # 选区半宽依然同上

fit_results = []  # 存放拟合可视化参数

print("\n>>> 开始进行局部精细化混合拟合（高斯拟合 + 抛物线顶点拟合）...")
print("-" * 75)

for i, center in enumerate(target_peaks):
    hw = half_widths[i]
    mask = (x_clean >= (center - hw)) & (x_clean <= (center + hw))
    x_sub = x_clean[mask]
    y_sub = y_pca1[mask]

    if len(x_sub) < 3:
        print(f"警告：目标点 {center} 附近数据点过少，跳过拟合。")
        fit_results.append(None)
        continue

    # =====================================================================
    # 前两个峰：采用原设定，使用非线性高斯拟合（带展宽）
    # =====================================================================
    if i < 2:
        init_guess = [max(y_sub), center, hw * 0.4]
        bounds = ([0, center - hw, 0.005], [np.inf, center + hw, hw * 2.0])
        try:
            popt, _ = curve_fit(
                gaussian, x_sub, y_sub, p0=init_guess, bounds=bounds
            )
            amp, mu, sigma = popt
            fwhm = 2.3548 * abs(sigma)

            fit_results.append(
                {"type": "gaussian", "popt": popt, "x_sub": x_sub, "hw": hw}
            )
            print(
                f"【峰 {i+1} 高斯拟合成功】(区间: {center-hw:.1f} ~ {center+hw:.1f})"
            )
            print(f"  -> 精确中心位置 (μ): {mu:.4f}")
            print(f"  -> 物理展宽 (FWHM):   {fwhm:.4f}")
        except Exception as e:
            print(f"【峰 {i+1} 高斯拟合失败】: {e}")
            fit_results.append(None)

    # =====================================================================
    # 第三个峰（16.3）：采用抛物线多项式拟合，100%稳健求峰位（无展宽）
    # =====================================================================
    else:
        try:
            # 拟合二次方程: y = ax^2 + bx + c
            poly_coefs = np.polyfit(x_sub, y_sub, 2)
            a, b, c = poly_coefs

            # 顶点横坐标公式: -b / (2a)
            mu_poly = -b / (2 * a)

            fit_results.append(
                {
                    "type": "polynomial",
                    "coefs": poly_coefs,
                    "x_sub": x_sub,
                    "hw": hw,
                    "mu": mu_poly,
                }
            )
            print(
                f"【峰 {i+1} 抛物线拟合成功】(区间: {center-hw:.2f} ~ {center+hw:.2f} | 差值: ±{hw})"
            )
            print(f"  -> 精确中心位置 (μ): {mu_poly:.4f} (无须计算展宽，稳定收敛)")
        except Exception as e:
            print(f"【峰 {i+1} 抛物线拟合失败】: {e}")
            fit_results.append(None)

print("-" * 75)

# =========================================================================
# 5. 可视化绘图展示
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=120)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --- 左图：全景 PCA1 波形与自定义选区阴影 ---
ax1.plot(x_clean, y_pca1, color="black", linewidth=2, label="PCA1 完整波形")
colors = ["purple", "green", "orange"]

for i, center in enumerate(target_peaks):
    hw = half_widths[i]
    ax1.axvspan(
        center - hw,
        center + hw,
        color=colors[i],
        alpha=0.15,
        label=f"峰 {i+1} 选区 ({center:.1f} ± {hw})",
    )

ax1.set_title("PCA1 全量波形与指定的三个局部选区", fontsize=12)
ax1.set_xlabel("角度", fontsize=10)
ax1.set_ylabel("标准幅值", fontsize=10)
ax1.grid(False)
ax1.legend(loc="upper right", fontsize=8)

# --- 右图：局部放大混合拟合曲线分解 ---
ax2.plot(x_clean, y_pca1, color="black", linewidth=1, alpha=0.3, label="基底信号")

for i, res in enumerate(fit_results):
    if res is not None:
        x_sub = res["x_sub"]
        hw = res["hw"]
        x_dense = np.linspace(x_sub.min(), x_sub.max(), 200)

        if res["type"] == "gaussian":
            popt = res["popt"]
            y_fit = gaussian(x_dense, *popt)
            fwhm_val = 2.3548 * abs(popt[2])
            label_str = (
                f"峰{i+1}(±{hw}) 高斯拟合 (中心:{popt[1]:.2f}, 展宽:{fwhm_val:.2f})"
            )
            ax2.plot(
                x_dense,
                y_fit,
                color=colors[i],
                linewidth=2.5,
                linestyle="--",
                label=label_str,
            )
            ax2.fill_between(x_dense, y_fit, alpha=0.3, color=colors[i])

        elif res["type"] == "polynomial":
            coefs = res["coefs"]
            mu_val = res["mu"]
            # 计算拟合出的抛物线 y 值
            y_fit = np.polyval(coefs, x_dense)
            label_str = f"峰{i+1}(±{hw}) 抛物线拟合 (精确中心:{mu_val:.2f})"
            ax2.plot(
                x_dense,
                y_fit,
                color=colors[i],
                linewidth=2.5,
                linestyle="--",
                label=label_str,
            )
            ax2.fill_between(ax2.get_xlim(), 0, 0, alpha=0)  # 仅占位保持色彩一致

ax2.set_title("各特征区域混合拟合结果 (峰3通过多项式求极值)", fontsize=12)
ax2.set_xlabel("角度", fontsize=10)
ax2.set_ylabel("标准幅值", fontsize=10)
ax2.grid(False)
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()