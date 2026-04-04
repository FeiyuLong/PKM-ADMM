# -*- coding: utf-8 -*-
"""
algorithms/saga_admm.py — Inexact SAGA-ADMM（方差缩减随机梯度 ADMM）

算法概述：
  SAGA 是 SAG 的改进版本，提供**无偏**的梯度估计（SAG 是有偏的）。
  SAGA 估计量利用当前样本梯度与梯度表中旧值的差分，加上梯度表均值作为控制变量，
  在理论上兼具低方差和无偏性。

变量定义（每个节点 m 维护，与 SAG-ADMM 相同的梯度表结构）：
  grad_table[m] : (n_m, d) 梯度表，第 j 行存储样本 j 的最新梯度
                  g_{m,j} = (1/M)·∇f_{m,j}(x_m)
  avg_grad[m]   : (d,) 梯度表行均值 = (1/n_m)·Σ_j g_{m,j}

算法伪代码（每个 epoch t = 0, 1, ..., T-1）：

  [Server]：
    y_{t+1} = S_{μ/(ρM)}((1/M)·Σ_m (x_{m,t} + λ_{m,t}))

  [节点 m 并行]：
    随机选取 j_t ∈ {0,...,n_m-1}
    计算当前梯度：g_new = (1/M)·∇f_{m,j_t}(x_{m,t})
    SAGA 梯度估计：
      v_{m,t+1} = g_new - g_{m,j_t}^old + avg_grad_m
                  （无偏：用新旧差分消除历史偏差，再加均值作为 baseline）
    更新梯度表：g_{m,j_t} ← g_new
    更新均值：avg_grad_m ← avg_grad_m + (g_new - g_old) / n_m
    x_{m,t+1} = x_{m,t} - (η/γ)·[v_{m,t+1} + ρ·(x_{m,t} - y_{t+1} + λ_{m,t})]
    λ_{m,t+1} = λ_{m,t} + x_{m,t+1} - y_{t+1}

SAGA vs SAG 对比：
  - SAG 用 avg_grad 直接作为梯度估计（有偏）
  - SAGA 用 g_new - g_old + avg_grad 作为梯度估计（无偏）
  - 两者梯度表维护逻辑完全相同，仅梯度估计量不同

与 Prompt 公式对应：
  v_{m,t+1} = (1/M)·(∇f_{m,j_t}(x_{m,t}) - g_{m,j_t}^t) + avg_grad_m
  其中 g_{m,j_t}^t = 梯度表中存储的旧值（已乘以 1/M）
"""

from typing import List

import numpy as np
import scipy.sparse as sp

from algorithms.base_admm import BaseADMM


class SagaADMM(BaseADMM):
    """
    SAGA-ADMM：基于 SAGA 无偏方差缩减梯度的分布式 ADMM。

    额外属性：
      self.grad_table : List[np.ndarray]，各节点梯度表（n_m × d）
      self.avg_grad   : List[np.ndarray]，各节点梯度表行均值（d,）
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
        eta: float = 0.005,
        gamma: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        """
        初始化 SAGA-ADMM。

        参数（继承自 BaseADMM 的参数见父类文档）
        ------
        eta : float
            基础学习率 η（默认 0.005）。
        gamma : float
            步长分母 γ，实际步长 = η/γ（默认 1.0）。
        batch_size : int
            每步采样的样本数，SAGA 标准设定为 1（默认 1）。
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
        self._step_size: float = eta / gamma

        # ---- 梯度表初始化（懒初始化为零）----
        # 与 SAG-ADMM 完全相同的存储结构
        self.grad_table: List[np.ndarray] = [
            np.zeros((self.X_nodes[m].shape[0], self.d), dtype=np.float64)
            for m in range(self.M)
        ]
        self.avg_grad: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]

    def step(self, t: int) -> None:
        """
        单步 SAGA-ADMM 迭代（t → t+1）。

        SAGA 与 SAG 的唯一区别：梯度估计量使用无偏的差分形式，
        而非直接使用梯度表均值。
        """
        # ---- Step 1：Server 更新 y ----
        self.y = self.compute_y_update()

        # ---- Step 2：各节点并行更新 ----
        for m in range(self.M):
            n_m: int = self.X_nodes[m].shape[0]
            bs: int = min(self.batch_size, n_m)

            # 随机选取样本下标
            idx: np.ndarray = np.random.choice(n_m, size=bs, replace=False)

            # SAGA 梯度估计：初始化为均值（baseline）
            # v_m = (1/bs)·Σ_{j∈idx} [g_new_j - g_old_j] + avg_grad_m
            v_m: np.ndarray = self.avg_grad[m].copy()

            for j in idx:
                # 当前样本 j 的新梯度（乘以 1/M）
                g_new: np.ndarray = self._single_sample_grad(m, j)
                g_old: np.ndarray = self.grad_table[m][j].copy()

                # SAGA 差分项：g_new - g_old（无偏修正）
                # 当 batch_size=1 时，差分直接加入；batch_size>1 时取平均
                v_m += (g_new - g_old) / bs

                # 更新梯度表均值（增量更新，避免全表求和）
                self.avg_grad[m] += (g_new - g_old) / n_m

                # 更新梯度表对应行
                self.grad_table[m][j] = g_new

            # ADMM 惩罚项
            admm_penalty: np.ndarray = self.rho * (
                self.x_nodes[m] - self.y + self.lam_nodes[m]
            )

            # x_m 更新
            self.x_nodes[m] = (
                self.x_nodes[m]
                - self._step_size * (v_m + admm_penalty)
            )

            # 对偶变量更新
            self.lam_nodes[m] = (
                self.lam_nodes[m] + self.x_nodes[m] - self.y
            )

    def _single_sample_grad(self, m: int, j: int) -> np.ndarray:
        """
        计算节点 m 中第 j 个样本的单样本梯度，乘以 (1/M)。

        返回：g_{m,j} = (1/M)·∇f_{m,j}(x_m)，shape (d,)
        """
        grad_local: np.ndarray = self.compute_local_grad(
            m, self.x_nodes[m], indices=np.array([j])
        )
        return grad_local / self.M
