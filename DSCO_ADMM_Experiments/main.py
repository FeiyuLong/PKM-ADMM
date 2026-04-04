# -*- coding: utf-8 -*-
"""
main.py — DSCO-ADMM 实验主程序入口

功能：
  遍历所有数据集（a9a, w8a, gisette），对每个数据集：
    1. 加载并预处理数据（特征缩放、标签规范化）
    2. 均匀划分到 M 个分布式节点
    3. 用 L-BFGS-B 精确计算全局最优值 F*
    4. 运行所有已注册的 ADMM 算法（各算法独立初始化，互不干扰）
    5. 生成并保存 5 张收敛曲线图

算法注册：
  在 ALGO_MAP 字典中添加 "算法名" → 算法类 的映射，
  main.py 会自动读取 ALGO_PARAMS 中对应的超参数并注入算法类构造函数。

图表输出：
  results/{dataset_name}/gap_vs_epoch.png
  results/{dataset_name}/gap_vs_time.png
  results/{dataset_name}/primal_vs_epoch.png
  results/{dataset_name}/dual_vs_epoch.png
  results/{dataset_name}/accuracy_vs_epoch.png

约束：
  【严格禁止】本程序完成代码生成后由开发者自行运行测试，
  不得在 AI Agent 会话期间调用 python main.py 或任何测试命令。
"""

import os
import sys
import time
from typing import Dict, List, Type

import numpy as np

# ---- 项目内部模块 ----
from config import GLOBAL_SETTINGS, DATASET_CONFIGS, ALGO_PARAMS, PLOT_CONFIG
from utils.data_utils import load_dataset, split_data_to_nodes, compute_node_stats
from utils.optimizer import compute_f_star
from utils.plot_utils import plot_all_metrics

# ---- 算法类导入 ----
from algorithms.base_admm import BaseADMM
from algorithms.stoc_admm import StocADMM
from algorithms.sag_admm import SagADMM
from algorithms.saga_admm import SagaADMM
from algorithms.svrg_admm import SvrgADMM
from algorithms.spider_admm import SpiderADMM
from algorithms.asvrg_admm import AsvrgADMM
from algorithms.pkm_admm import PkmADMM

# =========================================================================== #
#   算法注册表                                                                  #
# =========================================================================== #
#
# 格式：{"算法名称（与 ALGO_PARAMS 键名一致）": 算法类}
# 添加新算法时，在此注册并在 config.py 的 ALGO_PARAMS 中添加对应参数即可。

ALGO_MAP: Dict[str, Type[BaseADMM]] = {
    "STOC-ADMM":  StocADMM,
    "SAG-ADMM":   SagADMM,
    "SAGA-ADMM":  SagaADMM,
    "SVRG-ADMM":  SvrgADMM,
    "SPIDER-ADMM": SpiderADMM,
    "ASVRG-ADMM": AsvrgADMM,
    "PKM-ADMM":   PkmADMM,
}


# =========================================================================== #
#   单数据集实验流程                                                            #
# =========================================================================== #

def run_experiment(dataset_name: str) -> None:
    """
    对单个数据集运行所有已注册的 ADMM 算法并生成对比图表。

    参数
    ----
    dataset_name : str
        数据集名称，必须是 config.DATASET_CONFIGS 中的键之一。
    """
    print(f"\n{'=' * 65}")
    print(f"  数据集：{dataset_name}")
    print(f"{'=' * 65}")

    # ---- 读取全局配置 ----
    cfg: dict = DATASET_CONFIGS[dataset_name]
    M: int = GLOBAL_SETTINGS["M"]
    max_iter: int = GLOBAL_SETTINGS["max_iter"]
    mu: float = GLOBAL_SETTINGS["mu"]
    nu: float = GLOBAL_SETTINGS["nu"]
    rho: float = GLOBAL_SETTINGS["rho"]
    seed: int = GLOBAL_SETTINGS["seed"]

    # ---- 阶段 1：加载数据集 ----
    print(f"\n[1/4] 加载数据集...")
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_dataset(
        train_path=cfg["train_path"],
        test_path=cfg.get("test_path"),
        seed=seed,
    )
    print(f"      训练集: {X_train.shape[0]} 样本 × {X_train.shape[1]} 维  "
          f"（稀疏度 {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4%}）")
    print(f"      测试集: {X_test.shape[0]} 样本 × {X_test.shape[1]} 维")
    print(f"      耗时: {time.time() - t0:.2f}s")

    # ---- 阶段 2：划分到 M 个节点 ----
    print(f"\n[2/4] 均匀划分到 M={M} 个节点...")
    t0 = time.time()
    X_nodes, y_nodes = split_data_to_nodes(X_train, y_train, M, seed=seed)
    stats = compute_node_stats(X_nodes, y_nodes)
    n_per_node = stats["n_per_node"]
    print(f"      各节点样本数：min={min(n_per_node)}, "
          f"max={max(n_per_node)}, 平均={np.mean(n_per_node):.1f}")
    print(f"      耗时: {time.time() - t0:.2f}s")

    # ---- 阶段 3：计算全局最优值 F* ----
    print(f"\n[3/4] 用 L-BFGS-B 计算全局最优值 F*...")
    print(f"      （μ={mu}, ν={nu}；对于 gisette 可能需要几分钟）")
    t0 = time.time()
    F_star, x_star = compute_f_star(
        X_train=X_train,
        y_train=y_train,
        mu=mu,
        nu=nu,
        tol=1e-10,
        max_iter=5000,
        verbose=False,
    )
    print(f"      F* = {F_star:.8f}  （耗时 {time.time() - t0:.1f}s）")

    # ---- 阶段 4：运行各 ADMM 算法 ----
    print(f"\n[4/4] 运行 {len(ALGO_MAP)} 个 ADMM 算法（各 {max_iter} 个 Epoch）...")

    all_results: Dict[str, dict] = {}

    for algo_name, AlgoClass in ALGO_MAP.items():
        print(f"\n  ▶  [{algo_name}]")

        # 读取算法特有超参数（浅拷贝，防止修改原始配置）
        algo_params: dict = ALGO_PARAMS.get(algo_name, {}).copy()

        # 实例化算法（公共参数 + 算法特有参数）
        algo: BaseADMM = AlgoClass(
            X_nodes=X_nodes,
            y_nodes=y_nodes,
            X_test=X_test,
            y_test=y_test,
            mu=mu,
            nu=nu,
            rho=rho,
            F_star=F_star,
            max_iter=max_iter,
            seed=seed,
            **algo_params,   # 展开算法特有参数（eta, gamma, batch_size 等）
        )

        # 运行主循环
        t_start = time.time()
        history: dict = algo.run()
        t_total = time.time() - t_start

        all_results[algo_name] = history

        # 打印最终指标摘要
        final_gap = history["gap"][-1]
        final_primal = history["primal"][-1]
        final_dual = history["dual"][-1]
        final_acc = history["accuracy"][-1]
        print(f"     完成 {max_iter} epochs，总耗时 {t_total:.1f}s")
        print(f"     最终 Gap={final_gap:.4e}, Primal={final_primal:.4e}, "
              f"Dual={final_dual:.4e}, Acc={final_acc:.4f}")

    # ---- 生成并保存 5 张对比图 ----
    print(f"\n  生成对比图表...")
    plot_all_metrics(
        results=all_results,
        dataset_name=dataset_name,
        save_dir="results",
        figsize=tuple(PLOT_CONFIG["figsize"]),
        dpi=PLOT_CONFIG["dpi"],
        line_width=PLOT_CONFIG["line_width"],
    )
    print(f"  图表已保存到 results/{dataset_name}/")


# =========================================================================== #
#   主函数                                                                      #
# =========================================================================== #

def main() -> None:
    """
    程序主入口：遍历所有数据集运行 DSCO-ADMM 对比实验。
    """
    print("=" * 65)
    print("  DSCO-ADMM 分布式稀疏共识优化实验框架")
    print("=" * 65)
    print(f"  节点数 M       = {GLOBAL_SETTINGS['M']}")
    print(f"  最大迭代轮次   = {GLOBAL_SETTINGS['max_iter']}  （1 Epoch = 1 ADMM 迭代）")
    print(f"  L1 正则 μ      = {GLOBAL_SETTINGS['mu']}")
    print(f"  L2 正则 ν      = {GLOBAL_SETTINGS['nu']}")
    print(f"  ADMM 惩罚 ρ    = {GLOBAL_SETTINGS['rho']}")
    print(f"  随机种子       = {GLOBAL_SETTINGS['seed']}")
    print(f"  数据集         = {list(DATASET_CONFIGS.keys())}")
    print(f"  算法           = {list(ALGO_MAP.keys())}")

    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)

    # 遍历所有数据集
    datasets_to_run = list(DATASET_CONFIGS.keys())

    total_start = time.time()
    for dataset_name in datasets_to_run:
        run_experiment(dataset_name)

    print(f"\n\n{'=' * 65}")
    print(f"  所有实验完成！总耗时：{time.time() - total_start:.1f}s")
    print(f"  图表保存位置：results/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
