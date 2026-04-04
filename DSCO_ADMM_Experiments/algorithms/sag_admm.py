# -*- coding: utf-8 -*-
"""
algorithms/sag_admm.py — Inexact SAG-ADMM（随机平均梯度 ADMM）

算法概述：
  SAG（Stochastic Averaged Gradient）维护一张梯度表，每步仅随机选择
  一个样本更新对应行，用所有行的平均值作为本步梯度估计。
  相比纯随机梯度，SAG 的方差随迭代次数下降，收敛更稳定。

变量定义（每个节点 m 维护）：
  grad_table[m] : (n_m, d) 梯度表，第 j 行存储样本 j 的最新梯度
                  g_{m,j} = (1/M) · ∇f_{m,j}(x_m)（最后一次更新时的值）
  avg_grad[m]   : (d,) 梯度表的行平均 = (1/n_m)·Σ_j g_{m,j}

算法伪代码（每个 epoch t = 0, 1, ..., T-1）：

  [Server]：
    y_{t+1} = S_{μ/(ρM)}((1/M)·Σ_m (x_{m,t} + λ_{m,t}))

  [节点 m 并行]：
    均匀随机选取 j_t ∈ {0,...,n_m-1}
    更新梯度表：g_{m,j_t} ← (1/M)·∇f_{m,j_t}(x_{m,t})
    更新平均梯度：avg_grad_m ← avg_grad_m + (g_{m,j_t}^new - g_{m,j_t}^old) / n_m
    v_{m,t+1} = avg_grad_m                    ← SAG 梯度估计（有偏但低方差）
    x_{m,t+1} = x_{m,t} - (η/γ)·[v_{m,t+1} + ρ·(x_{m,t} - y_{t+1} + λ_{m,t})]
    λ_{m,t+1} = λ_{m,t} + x_{m,t+1} - y_{t+1}

注意事项：
  1. 梯度表初始化：g_{m,j} = (1/M)·∇f_{m,j}(x_{m,0})（x_0=0，梯度为常数）。
     为避免初始化时的大规模计算（对 gisette 等大数据集很慢），
     将梯度表懒初始化为零向量，等效于假设初始梯度为 0。
     这会引入初始偏差，但随着样本被访问，偏差迅速消除。
  2. 与 Prompt 公式的对应：
     Prompt 中 g_{m,j_t} 存储的是 (1/M)·∇f_{m,j_t}(x_m)，
     本实现中梯度表也乘以 (1/M) 因子，与公式完全一致。
  3. SAG 是有偏估计（与 SAGA 的无偏估计对比），理论保证需要步长足够小。
"""

from typing import List

import numpy as np
import scipy.sparse as sp

from algorithms.base_admm import BaseADMM


class SagADMM(BaseADMM):
    """
    SAG-ADMM：基于随机平均梯度的分布式 ADMM。

    额外属性：
      self.grad_table  : List[np.ndarray]，各节点的梯度表
                         grad_table[m].shape = (n_m, d)，密集存储
      self.avg_grad    : List[np.ndarray]，各节点梯度表的行均值
                         avg_grad[m].shape = (d,)
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
        初始化 SAG-ADMM。

        参数（继承自 BaseADMM 的参数见父类文档）
        ------
        eta : float
            基础学习率 η，SAG 通常需要较小步长（默认 0.005）。
        gamma : float
            步长分母 γ，实际步长为 η/γ（默认 1.0）。
        batch_size : int
            每步随机选取的样本数，SAG 理论上每步选 1 个（默认 1）。
            batch_size > 1 时为 mini-batch SAG，随机选取 batch_size 个样本
            并各自更新对应梯度表行（加速初始收敛）。
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
        # grad_table[m][j] = (1/M)·∇f_{m,j}(x_m)，第 j 个样本的最新梯度
        # 初始化为零等效于假设 x_m=0 时梯度为 0（会被快速纠正）
        # 注意：对于大数据集（gisette: 600×5000），直接存储稠密梯度表
        # 内存占用约 600×5000×8 bytes ≈ 24 MB / 节点，可接受
        self.grad_table: List[np.ndarray] = [
            np.zeros((self.X_nodes[m].shape[0], self.d), dtype=np.float64)
            for m in range(self.M)
        ]
        # avg_grad[m] = (1/n_m)·Σ_j grad_table[m][j]
        # 初始时全零（与 grad_table 全零一致）
        self.avg_grad: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]

    def step(self, t: int) -> None:
        """
        单步 SAG-ADMM 迭代（t → t+1）。

        执行顺序：
          1. Server：软阈值更新 y_{t+1}
          2. 节点 m：随机选样本 → 增量更新梯度表 → SAG 梯度估计 → 更新 x_m, λ_m
        """
        # ---- Step 1：Server 更新 y ----
        self.y = self.compute_y_update()

        # ---- Step 2：各节点并行更新 ----
        for m in range(self.M):
            n_m: int = self.X_nodes[m].shape[0]
            bs: int = min(self.batch_size, n_m)

            # 随机选取样本下标（可重复，SAG 标准设定）
            idx: np.ndarray = np.random.choice(n_m, size=bs, replace=False)

            for j in idx:
                # 计算样本 j 的新梯度（单样本梯度，包含 L2 正则项）
                # compute_local_grad 的 indices=[j] 等价于单样本梯度
                g_new: np.ndarray = self._single_sample_grad(m, j)

                # 增量更新均值：avg ← avg + (g_new - g_old) / n_m
                g_old: np.ndarray = self.grad_table[m][j]
                self.avg_grad[m] += (g_new - g_old) / n_m

                # 更新梯度表第 j 行
                self.grad_table[m][j] = g_new

            # SAG 梯度估计 v_{m,t+1} = avg_grad_m（梯度表行均值）
            v_m: np.ndarray = self.avg_grad[m]

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
        计算节点 m 中第 j 个样本处的单样本梯度乘以 (1/M) 系数。

        梯度定义（与梯度表存储格式一致）：
            g_{m,j} = (1/M)·∇f_{m,j}(x_m)
                    = (1/M)·[ -b_{m,j}·σ(-b_{m,j}·a_{m,j}^T·x_m)·a_{m,j}
                               + ν·x_m ]

        注意：这里不除以 n_m（单样本梯度，非批均值），
        与梯度表行均值 avg_grad = (1/n_m)·Σ_j g_{m,j} 配合使用。

        参数
        ----
        m : int，节点编号
        j : int，样本编号（节点内下标）

        返回
        ----
        numpy.ndarray, shape (d,)
        """
        # 调用基类 compute_local_grad，传入单样本下标
        # compute_local_grad 返回 (1/|I|)·∂loss/∂x + ν·x
        # 单样本时 |I|=1，结果即为单样本梯度
        grad_local: np.ndarray = self.compute_local_grad(
            m, self.x_nodes[m], indices=np.array([j])
        )
        # 乘以 (1/M) 因子（与 Prompt 公式 g_{m,j} = (1/M)·∇f_{m,j} 一致）
        return grad_local / self.M
