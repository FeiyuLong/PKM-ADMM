# -*- coding: utf-8 -*-
"""
algorithms/stoc_admm.py — Inexact STOC-ADMM（分布式随机 ADMM）

算法来源：Huang et al., "Stochastic Consensus ADMM for Distributed Learning"

数学描述（每个 epoch t = 0, 1, ..., T-1）：

  [Server]：
    ū_t = (1/M)·Σ_m (x_{m,t} + λ_{m,t})
    y_{t+1} = S_{μ/(ρM)}(ū_t)                       ← 软阈值更新全局稀疏变量

  [节点 m 并行（m = 1,...,M）]：
    采样 mini-batch I_{m,t} ⊂ D_m，大小 = batch_size
    v_{m,t+1} = (1/|I|)·Σ_{i∈I} ∇f_{m,i}(x_{m,t})  ← 局部随机梯度估计
    x_{m,t+1} = x_{m,t} - (η/γ)·[v_{m,t+1} + ρ·(x_{m,t} - y_{t+1} + λ_{m,t})]
    λ_{m,t+1} = λ_{m,t} + x_{m,t+1} - y_{t+1}       ← 对偶变量更新（缩放形式）

说明：
  - v_{m,t+1} 是对 (1/n_m)·Σ_{i∈D_m} ∇f_{m,i}(x_{m,t}) 的无偏随机估计。
  - x 更新步骤中的 ADMM 惩罚项为 ρ·(x_m - y + λ_m)，这是标准 Scaled ADMM 形式。
  - 与 Prompt 中公式的对应：v_{m,t+1} 使用局部平均（(1/|I|)），
    而非全局平均（(1/(M·|I|))）。两者在 M 节点均衡时通过 ρ 的缩放等价。
  - γ=1 时步长即为 η；通过 γ 可以在保持 η 不变的情况下降低有效步长。
"""

from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from algorithms.base_admm import BaseADMM


class StocADMM(BaseADMM):
    """
    STOC-ADMM：基于 mini-batch 随机梯度的分布式 ADMM。

    继承 BaseADMM，仅需实现 step() 方法。
    """

    def __init__(
        self,
        X_nodes: List[sp.csr_matrix],
        y_nodes: List[np.ndarray],
        X_test: sp.csr_matrix,
        y_test: np.ndarray,
        mu: float,
        nu: float,
        rho: float,
        F_star: float,
        max_iter: int,
        seed: int = 42,
        eta: float = 0.01,
        gamma: float = 1.0,
        batch_size: int = 32,
    ) -> None:
        """
        初始化 STOC-ADMM。

        参数（继承自 BaseADMM 的参数见父类 __init__ 文档）
        ------
        eta : float
            基础学习率 η（步长分子），默认 0.01。
        gamma : float
            步长分母参数 γ，实际步长为 η/γ，默认 1.0（即步长 = η）。
        batch_size : int
            每节点每轮的 mini-batch 大小，默认 32。
            若节点样本数 n_m < batch_size，则使用全部样本。
        """
        super().__init__(
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
        )
        self.eta: float = eta
        self.gamma: float = gamma
        self.batch_size: int = batch_size

        # 有效步长：η/γ
        self._step_size: float = eta / gamma

    def step(self, t: int) -> None:
        """
        单步 STOC-ADMM 迭代（t → t+1）。

        执行顺序：
          1. Server 聚合并软阈值更新 y_{t+1}
          2. 各节点采样 mini-batch，计算随机梯度，更新 x_m 和 λ_m

        参数
        ----
        t : int
            当前迭代轮次（0-indexed），暂未使用（预留供自适应步长等扩展）。
        """
        # ---- Step 1：Server 更新全局共识变量 y ----
        self.y = self.compute_y_update()

        # ---- Step 2：各节点并行更新（Python 用循环模拟并行）----
        for m in range(self.M):
            n_m: int = self.X_nodes[m].shape[0]
            bs: int = min(self.batch_size, n_m)

            # 无放回随机采样 mini-batch 下标
            idx: np.ndarray = np.random.choice(n_m, size=bs, replace=False)

            # 计算节点 m 的局部随机梯度估计 v_{m,t+1}
            # compute_local_grad 已包含 L2 正则梯度 ν·x_m
            v_m: np.ndarray = self.compute_local_grad(
                m, self.x_nodes[m], indices=idx
            )

            # ADMM 惩罚项：ρ·(x_{m,t} - y_{t+1} + λ_{m,t})
            # 注：此处 y 已更新为 y_{t+1}
            admm_penalty: np.ndarray = self.rho * (
                self.x_nodes[m] - self.y + self.lam_nodes[m]
            )

            # x_m 梯度下降更新
            self.x_nodes[m] = (
                self.x_nodes[m]
                - self._step_size * (v_m + admm_penalty)
            )

            # 缩放对偶变量更新（Scaled Dual Update）
            # λ_{m,t+1} = λ_{m,t} + x_{m,t+1} - y_{t+1}
            self.lam_nodes[m] = (
                self.lam_nodes[m] + self.x_nodes[m] - self.y
            )
