# -*- coding: utf-8 -*-
"""
utils/plot_utils.py — 收敛曲线绘制与保存

功能：
  对每个数据集，生成 5 张收敛曲线图：
    图1：横坐标 Epoch，纵坐标 Optimality Gap（对数刻度）
    图2：横坐标 Time (s)，纵坐标 Optimality Gap（对数刻度）
    图3：横坐标 Epoch，纵坐标 Primal Residual（对数刻度）
    图4：横坐标 Epoch，纵坐标 Dual Residual（对数刻度）
    图5：横坐标 Epoch，纵坐标 Test Accuracy（线性刻度）

图片保存到 results/{dataset_name}/ 目录，文件名见 FILENAME_MAP。

字体：使用跨平台安全字体 DejaVu Sans，避免 Windows 中文字体缺失导致乱码。
"""

import os
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 在无 GUI 环境下使用非交互式后端（避免 Tkinter 报错）
matplotlib.use("Agg")

# 跨平台字体配置
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# =========================================================================== #
#   常量：文件名映射                                                            #
# =========================================================================== #

FILENAME_MAP: Dict[str, str] = {
    "gap_epoch":      "gap_vs_epoch.png",
    "gap_time":       "gap_vs_time.png",
    "primal_epoch":   "primal_vs_epoch.png",
    "dual_epoch":     "dual_vs_epoch.png",
    "accuracy_epoch": "accuracy_vs_epoch.png",
}


# =========================================================================== #
#   公共接口                                                                    #
# =========================================================================== #

def plot_all_metrics(
    results: Dict[str, Dict[str, List[float]]],
    dataset_name: str,
    save_dir: str = "results",
    figsize: tuple = (8, 5),
    dpi: int = 150,
    line_width: float = 1.8,
) -> None:
    """
    为单个数据集生成并保存 5 张收敛曲线图。

    参数
    ----
    results : dict
        键为算法名称（str），值为 BaseADMM.history 字典，包含：
          "gap"      : List[float]，每 epoch 的 Optimality Gap
          "primal"   : List[float]，每 epoch 的 Primal Residual
          "dual"     : List[float]，每 epoch 的 Dual Residual
          "accuracy" : List[float]，每 epoch 的 Test Accuracy
          "time"     : List[float]，每 epoch 结束时的累计 Wall-clock 时间（秒）
    dataset_name : str
        数据集名称，用于子目录命名和图标题。
    save_dir : str
        图片保存根目录，默认为 "results"。
    figsize : tuple
        单张图尺寸（宽, 高），单位英寸。
    dpi : int
        图片分辨率（像素/英寸）。
    line_width : float
        曲线线宽。
    """
    out_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    algo_names = list(results.keys())

    # ---- 图1：Optimality Gap vs Epoch ----
    _save_single_plot(
        results=results,
        algo_names=algo_names,
        metric_key="gap",
        x_key=None,                  # x 轴使用 epoch 编号（1-indexed）
        xlabel="Epoch",
        ylabel="Optimality Gap  |F - F*|",
        title=f"Optimality Gap vs Epoch  [{dataset_name}]",
        save_path=os.path.join(out_dir, FILENAME_MAP["gap_epoch"]),
        log_y=True,
        figsize=figsize,
        dpi=dpi,
        lw=line_width,
    )

    # ---- 图2：Optimality Gap vs Time ----
    _save_single_plot(
        results=results,
        algo_names=algo_names,
        metric_key="gap",
        x_key="time",                # x 轴使用累计时间
        xlabel="Wall-clock Time (s)",
        ylabel="Optimality Gap  |F - F*|",
        title=f"Optimality Gap vs Time  [{dataset_name}]",
        save_path=os.path.join(out_dir, FILENAME_MAP["gap_time"]),
        log_y=True,
        figsize=figsize,
        dpi=dpi,
        lw=line_width,
    )

    # ---- 图3：Primal Residual vs Epoch ----
    _save_single_plot(
        results=results,
        algo_names=algo_names,
        metric_key="primal",
        x_key=None,
        xlabel="Epoch",
        ylabel="Primal Residual  sqrt(1/M * sum ||x_m - y||^2)",
        title=f"Primal Residual vs Epoch  [{dataset_name}]",
        save_path=os.path.join(out_dir, FILENAME_MAP["primal_epoch"]),
        log_y=True,
        figsize=figsize,
        dpi=dpi,
        lw=line_width,
    )

    # ---- 图4：Dual Residual vs Epoch ----
    _save_single_plot(
        results=results,
        algo_names=algo_names,
        metric_key="dual",
        x_key=None,
        xlabel="Epoch",
        ylabel="Dual Residual  rho * ||y_t - y_{t-1}||",
        title=f"Dual Residual vs Epoch  [{dataset_name}]",
        save_path=os.path.join(out_dir, FILENAME_MAP["dual_epoch"]),
        log_y=True,
        figsize=figsize,
        dpi=dpi,
        lw=line_width,
    )

    # ---- 图5：Test Accuracy vs Epoch ----
    _save_single_plot(
        results=results,
        algo_names=algo_names,
        metric_key="accuracy",
        x_key=None,
        xlabel="Epoch",
        ylabel="Test Accuracy",
        title=f"Test Accuracy vs Epoch  [{dataset_name}]",
        save_path=os.path.join(out_dir, FILENAME_MAP["accuracy_epoch"]),
        log_y=False,               # 准确率使用线性刻度
        figsize=figsize,
        dpi=dpi,
        lw=line_width,
    )


# =========================================================================== #
#   私有辅助函数                                                                #
# =========================================================================== #

def _save_single_plot(
    results: Dict[str, Dict[str, List[float]]],
    algo_names: List[str],
    metric_key: str,
    x_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    log_y: bool,
    figsize: tuple,
    dpi: int,
    lw: float,
) -> None:
    """
    绘制并保存单张收敛曲线图。

    参数
    ----
    results : dict
        同 plot_all_metrics 中的 results。
    algo_names : list[str]
        所有算法名称列表，控制图例顺序。
    metric_key : str
        纵轴指标的键名（如 "gap", "primal", "dual", "accuracy"）。
    x_key : str or None
        横轴数据的键名。若为 None，则使用 Epoch 序号（从 1 开始）。
        若为 "time"，则使用累计时间（秒）。
    xlabel, ylabel, title : str
        坐标轴标签和图标题。
    save_path : str
        图片保存的完整文件路径。
    log_y : bool
        是否对纵轴使用对数刻度。
    figsize, dpi, lw : 图形参数。
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in algo_names:
        hist = results[name]
        ys = np.array(hist[metric_key], dtype=np.float64)

        if x_key is None:
            # Epoch 序号（1-indexed）
            xs = np.arange(1, len(ys) + 1, dtype=np.float64)
        else:
            xs = np.array(hist[x_key], dtype=np.float64)

        # ---- 处理对数刻度下的非正值（替换为极小正值以避免警告）----
        if log_y:
            ys_plot = np.where(ys > 0, ys, 1e-15)
        else:
            ys_plot = ys

        ax.plot(xs, ys_plot, label=name, linewidth=lw)

    # ---- 坐标轴配置 ----
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best", fontsize=10, framealpha=0.8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"    [plot_utils] 已保存：{save_path}")
