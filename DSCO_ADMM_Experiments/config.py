# -*- coding: utf-8 -*-
"""
config.py — 超参数配置中心

集中管理 DSCO-ADMM 实验的所有超参数，包括：
  - 数据集路径与格式描述
  - 全局实验设置（节点数、迭代次数、正则系数等）
  - 各算法独立的可调参数字典
  - 绘图配置

设计原则：
  main.py 从此处读取所有配置，算法类通过关键字参数接收超参数，
  修改超参数时只需改动本文件，无需触碰算法实现。
"""

# =========================================================================== #
#   1. 数据集路径配置                                                           #
# =========================================================================== #

# 数据集位于 DSCO_ADMM_Experiments/datasets/ 目录下
# sklearn.datasets.load_svmlight_file 原生支持 .bz2 压缩格式，无需手动解压
DATASET_CONFIGS: dict = {
    "a9a": {
        "train_path": "datasets/a9a",
        "test_path":  "datasets/a9a.t",
        "format":     "libsvm",          # 标准 libsvm 格式
    },
    "w8a": {
        "train_path": "datasets/w8a",
        "test_path":  "datasets/w8a.t",
        "format":     "libsvm",
    },
    "gisette": {
        "train_path": "datasets/gisette_scale.bz2",
        "test_path":  "datasets/gisette_scale.t.bz2",
        "format":     "libsvm_bz2",      # bz2 压缩 libsvm 格式
    },
}

# =========================================================================== #
#   2. 全局实验设置                                                             #
# =========================================================================== #

GLOBAL_SETTINGS: dict = {
    # 分布式节点数 M
    "M":        10,

    # 最大迭代轮次（1 Epoch = 1 ADMM 迭代）
    "max_iter": 100,

    # L1 稀疏正则系数 μ（对应全局目标中的 μ‖y‖₁ 项）
    "mu":       1e-3,

    # L2 正则系数 ν（对应各节点局部损失中的 (ν/2)‖x_m‖² 项）
    "nu":       1e-3,

    # ADMM 惩罚参数 ρ（各算法共用初始值，部分算法可在 ALGO_PARAMS 中覆盖）
    "rho":      1.0,

    # 随机种子，确保实验可复现
    "seed":     42,
}

# =========================================================================== #
#   3. 各算法独立超参数                                                         #
# =========================================================================== #
#
# 注意：mu, nu, rho, M, max_iter, seed 均由 GLOBAL_SETTINGS 提供给算法类；
#       此处仅存储各算法特有的参数，通过 **kwargs 展开注入算法类 __init__。
#
# 参数含义说明（通用）：
#   eta        — 基础学习率（步长 η）
#   gamma      — Inexact 步长分母参数（γ），使实际步长为 η/γ
#   batch_size — 每节点每轮 mini-batch 大小
#
# 参数含义说明（特有）：
#   update_freq — SVRG/SPIDER 快照更新频率（每隔多少步重算全梯度）
#   inner_iter  — ASVRG 内层循环步数 E
#   theta       — ASVRG 加速动量插值系数 θ
#   tau         — PKM Katyusha 动量参数 τ
#   varrho      — PKM Katyusha 动量参数 ρ'（区别于 ADMM 惩罚参数 ρ）
#   update_prob — PKM 快照更新概率 p_t

ALGO_PARAMS: dict = {
    "STOC-ADMM": {
        "eta":        0.01,
        "gamma":      1.0,
        "batch_size": 32,
    },
    "SAG-ADMM": {
        "eta":        0.005,
        "gamma":      1.0,
        # SAG 每步仅采样单样本（batch_size=1 即随机单样本 SAG）
        "batch_size": 1,
    },
    "SAGA-ADMM": {
        "eta":        0.005,
        "gamma":      1.0,
        "batch_size": 1,
    },
    "SVRG-ADMM": {
        "eta":         0.01,
        "gamma":       1.0,
        "batch_size":  32,
        # 内层循环长度 K（每 K 步后更新一次全梯度快照）
        "update_freq": 20,
    },
    "SPIDER-ADMM": {
        "eta":         0.02,
        "gamma":       1.0,
        "batch_size":  32,
        # 每隔 q 步计算一次完整全梯度
        "update_freq": 50,
    },
    "ASVRG-ADMM": {
        "eta":        0.01,
        "gamma":      1.0,
        "batch_size": 32,
        # 内层循环步数 E
        "inner_iter": 10,
        # Katyusha 加速动量插值系数 θ ∈ (0, 1]
        "theta":      0.48,
    },
    "PKM-ADMM": {
        "eta":         0.01,
        "gamma":       1.0,
        "batch_size":  32,
        # Katyusha 动量参数：τ + varrho ≤ 1
        "tau":         0.5,
        "varrho":      0.3,
        # 快照变量 w_m 的随机更新概率
        "update_prob": 0.1,
    },
}

# =========================================================================== #
#   4. 绘图配置                                                                 #
# =========================================================================== #

PLOT_CONFIG: dict = {
    "figsize":     (8, 5),          # 单张图尺寸（宽×高，英寸）
    "dpi":         150,             # 分辨率
    # 使用跨平台安全字体，避免 Windows 中文字体缺失问题
    "font_family": "DejaVu Sans",
    "log_scale":   True,            # Gap / Residual 图使用对数刻度
    "save_format": "png",           # 输出格式
    "line_width":  1.8,             # 曲线线宽
}
