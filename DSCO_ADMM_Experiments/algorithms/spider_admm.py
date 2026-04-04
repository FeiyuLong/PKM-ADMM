# -*- coding: utf-8 -*-
"""
algorithms/spider_admm.py — Inexact SPIDER-ADMM（递推差分方差缩减 ADMM）

算法概述：
  SPIDER（Stochastically Path-Integrated Differential EstimatoR）是一种
  递推式方差缩减梯度估计器，无需存储梯度表（节省内存）。
  它通过维护一个梯度估计向量 v_m，在每步用相邻两次迭代点的梯度差分增量更新，
  仅在每隔 q 步时重置为完整全梯度。

变量定义（每个节点 m 维护）：
  v_m[m]     : (d,) SPIDER 梯度估计向量（递推维护）
  x_prev[m]  : (d,) 前一步的 x_m（用于计算梯度差分）

算法伪代码（每个 epoch t = 0, 1, ..., T-1）：

  [Server]：
    y_{t+1} = S_{μ/(ρM)}((1/M)·Σ_m (x_{m,t} + λ_{m,t}))

  [节点 m 并行]：
    若 t % q == 0（外层循环起点）：
      v_{m,t+1} = (1/M)·∇f_m(x_{m,t})          ← 重置为完整全梯度
    否则（递推更新）：
      采样 batch J, |J|=b
      v_{m,t+1} = (1/(Mb))·Σ_{j∈J} [∇f_{m,j}(x_{m,t}) - ∇f_{m,j}(x_{m,t-1})] + v_{m,t}
                ← 用相邻点差分增量修正 v_m，不改变其方差量级
    x_{m,t+1} = x_{m,t} - (η/γ)·[v_{m,t+1} + ρ·(x_{m,t} - y_{t+1} + λ_{m,t})]
    x_prev_m = x_{m,t}                           ← 保存本步 x_m 供下步使用
    λ_{m,t+1} = λ_{m,t} + x_{m,t+1} - y_{t+1}

Epoch 计数：
  每次调用 step(t) 执行一次完整迭代。当 t % q == 0 时需计算全梯度（较慢）。

SPIDER vs SVRG 对比：
  - SVRG：每 K 步计算全梯度，其余步用单样本差分
  - SPIDER：每 q 步重置全梯度，其余步用批差分（x_t - x_{t-1}），
            差分项的方差会随迭代收敛而下降（路径积分特性）
  - SPIDER 通常比 SVRG 更高效，但实现更复杂

与 Prompt 公式对应：
  全梯度步：v_{m,t+1} = (1/M)·∇f_m(x_{m,t})
  递推步：  v_{m,t+1} = (1/(M·b))·Σ_{j∈J} [∇f_{m,j}(x_{m,t}) - ∇f_{m,j}(x_{m,t-1})] + v_{m,t}
  本实现中 compute_local_grad 的 1/n_m 因子与 (1/b) 因子来自 base 类，
  全梯度时不除以 M（base 类返回的是 ∇f_m，再除以 M）；
  递推步的差分梯度同样需要除以 M（与 Prompt 中 1/(M·b) 对应）。
"""

from typing import List

import numpy as np
import scipy.sparse as sp

from algorithms.base_admm import BaseADMM


class SpiderADMM(BaseADMM):
    """
    SPIDER-ADMM：基于递推路径差分方差缩减的分布式 ADMM。

    额外属性：
      self.v_nodes    : List[np.ndarray]，各节点 SPIDER 梯度估计向量
      self.x_prev     : List[np.ndarray]，各节点前一步的 x_m（用于差分）
      self.update_freq: int，全梯度重置周期 q
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
        eta: float = 0.02,
        gamma: float = 1.0,
        batch_size: int = 32,
        update_freq: int = 50,
    ) -> None:
        """
        初始化 SPIDER-ADMM。

        参数（继承自 BaseADMM 的参数见父类文档）
        ------
        eta : float
            基础学习率 η（默认 0.02，SPIDER 通常可用较大步长）。
        gamma : float
            步长分母 γ（默认 1.0）。
        batch_size : int
            递推差分估计的 mini-batch 大小 b（默认 32）。
        update_freq : int
            全梯度重置周期 q（默认 50）。每隔 q 步计算一次完整全梯度。
            较小的 q 更新更频繁（精度高但计算量大），
            较大的 q 减少全梯度计算（但估计偏差可能积累）。
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
        self.update_freq: int = update_freq
        self._step_size: float = eta / gamma

        # ---- SPIDER 状态变量初始化 ----
        # v_m：梯度估计向量，初始化为 x_m=0 处的全梯度 (1/M)·∇f_m(0)
        self.v_nodes: List[np.ndarray] = [
            self.compute_local_grad(m, np.zeros(self.d)) / self.M
            for m in range(self.M)
        ]
        # x_prev[m]：前一步的 x_m，初始化为零（与 x_m 初始值相同）
        self.x_prev: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]

    def step(self, t: int) -> None:
        """
        单步 SPIDER-ADMM 迭代（t → t+1）。

        执行顺序：
          1. Server：更新 y_{t+1}
          2. 节点 m：
             - 若 t % q == 0：计算完整全梯度重置 v_m
             - 否则：用 batch 差分增量更新 v_m
             - 用 v_m 执行 x_m 更新
             - 保存 x_m 为 x_prev（供下步使用）
             - 更新 λ_m

        参数
        ----
        t : int
            当前 epoch 编号（0-indexed）。
        """
        # ---- Step 1：Server 更新 y ----
        self.y = self.compute_y_update()

        # ---- Step 2：各节点并行更新 ----
        for m in range(self.M):
            n_m: int = self.X_nodes[m].shape[0]
            x_m_cur: np.ndarray = self.x_nodes[m]   # 当前 x_{m,t}

            if t % self.update_freq == 0:
                # ---- 全梯度重置：v_{m,t+1} = (1/M)·∇f_m(x_{m,t}) ----
                # compute_local_grad 返回 ∇f_m（对全部样本），除以 M 后存入 v_m
                v_m_new: np.ndarray = (
                    self.compute_local_grad(m, x_m_cur) / self.M
                )
            else:
                # ---- 递推差分更新 ----
                # v_{m,t+1} = (1/(M·b))·Σ_{j∈J} [∇f_{m,j}(x_t) - ∇f_{m,j}(x_{t-1})] + v_t
                bs: int = min(self.batch_size, n_m)
                idx: np.ndarray = np.random.choice(n_m, size=bs, replace=False)

                # 在 x_{m,t} 和 x_{m,t-1} 处分别计算 mini-batch 梯度
                grad_cur: np.ndarray = self.compute_local_grad(
                    m, x_m_cur, indices=idx
                )
                grad_prev: np.ndarray = self.compute_local_grad(
                    m, self.x_prev[m], indices=idx
                )

                # 差分项除以 M（与 Prompt 中 1/(M·b) 的 M 因子对应）
                # base 类的 compute_local_grad 已除以 batch_size（即 b），
                # 这里再除以 M 得到 (1/(M·b)) 系数
                diff: np.ndarray = (grad_cur - grad_prev) / self.M

                # 递推更新 v_m
                v_m_new = self.v_nodes[m] + diff

            # ---- 保存本步 x_m（供下一步差分使用，在 x_m 更新前保存）----
            self.x_prev[m] = x_m_cur.copy()

            # ---- 更新 SPIDER 梯度估计 ----
            self.v_nodes[m] = v_m_new

            # ---- x_m 更新 ----
            admm_penalty: np.ndarray = self.rho * (
                x_m_cur - self.y + self.lam_nodes[m]
            )
            self.x_nodes[m] = (
                x_m_cur
                - self._step_size * (v_m_new + admm_penalty)
            )

            # ---- 对偶变量更新 ----
            self.lam_nodes[m] = (
                self.lam_nodes[m] + self.x_nodes[m] - self.y
            )
