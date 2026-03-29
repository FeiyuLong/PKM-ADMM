import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 解决Windows字体警告 + 负号显示问题
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 自动创建保存目录
os.makedirs("results", exist_ok=True)

def plot_convergence_curves_by_epoch(results, algo_names):
    # 目标函数间隙
    plt.figure()
    for res, name in zip(results, algo_names):
        plt.plot(res["gap"], label=name)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Optimality Gap " + r"$|F - F^*|$")
    plt.title("Objective Function Gap Convergence Curve vs Epoch")
    # plt.legend()
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig("results/gap_curve_epoch.png", format="png")
    plt.close()

    # 原始残差
    plt.figure()
    for res, name in zip(results, algo_names):
        plt.plot(res["primal"], label=name)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Primal Residual " + r"$ \| D \mathbf{x}_{t} - \mathbf{y}_{t} \|$")
    plt.title("Primal Residual Convergence Curve vs Epoch")
    # plt.legend()
    plt.legend(loc='center right') 
    plt.grid(True)
    plt.savefig("results/primal_curve_epoch.png", format="png")
    plt.close()

    # 对偶残差
    plt.figure()
    for res, name in zip(results, algo_names):
        plt.plot(res["dual"], label=name)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Dual Residual " + r"$\| \boldsymbol{\lambda}_{t} - \boldsymbol{\lambda}^* \|$")
    plt.title("Dual Residual Convergence Curve vs. Epochs")
    # plt.legend()
    plt.legend(loc='center right') 
    plt.grid(True)
    plt.savefig("results/dual_curve_epoch.png", format="png")
    plt.close()


def plot_convergence_curves_by_time(results, algo_names, save_dir="results"):
    """
    按运行时间绘制收敛曲线（统一x轴长度）
    分别保存3张独立图片：gap / primal / dual
    """
    os.makedirs(save_dir, exist_ok=True)

    # 核心修改：计算所有算法的总时间，取最短的总时间作为时间上限
    total_times = [res["time"][-1] for res in results]  # 每个算法跑完max_iter的总时间
    max_time_limit = min(total_times)  # 最短总时间作为锚点

    # ===================== 1. 目标函数间隙 vs 时间 =====================
    plt.figure()
    for res, name in zip(results, algo_names):
        # 将列表转换为numpy数组，支持向量化比较
        time_arr = np.array(res["time"])
        mask = time_arr <= max_time_limit
        filtered_time = time_arr[mask]
        filtered_gap = np.array(res["gap"])[mask]
        plt.plot(filtered_time, filtered_gap, label=name)
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimality Gap " + r"$|F - F^*|$")
    plt.title("Objective Function Gap vs Time")
    # plt.legend()
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "gap_curve_time.png"), format="png")
    plt.close()

    # ===================== 2. 原始残差 vs 时间 =====================
    plt.figure()
    for res, name in zip(results, algo_names):
        # 将列表转换为numpy数组，支持向量化比较
        time_arr = np.array(res["time"])
        mask = time_arr <= max_time_limit
        filtered_time = time_arr[mask]
        filtered_primal = np.array(res["primal"])[mask]
        plt.plot(filtered_time, filtered_primal, label=name)
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Primal Residual " + r"$ \| D \mathbf{x}_{t} - \mathbf{y}_{t} \|$")
    plt.title("Primal Residual vs Time")
    # plt.legend()
    plt.legend(loc='center right') 
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "primal_curve_time.png"), format="png")
    plt.close()

    # ===================== 3. 对偶残差 vs 时间 =====================
    plt.figure()
    for res, name in zip(results, algo_names):
        # 将列表转换为numpy数组，支持向量化比较
        time_arr = np.array(res["time"])
        mask = time_arr <= max_time_limit
        filtered_time = time_arr[mask]
        filtered_dual = np.array(res["dual"])[mask]
        plt.plot(filtered_time, filtered_dual, label=name)
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Dual Residual " + r"$\| \boldsymbol{\lambda}_{t} - \boldsymbol{\lambda}^* \|$")
    plt.title("Dual Residual vs Time")
    # plt.legend()
    plt.legend(loc='center right') 
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "dual_curve_time.png"), format="png")
    plt.close()