# -*- coding: utf-8 -*-
"""
algorithms/pkm_admm.py — Inexact PKM-ADMM（近端 Katyusha 动量 ADMM）

算法来源：DSCO 框架中最复杂的随机 ADMM 变体，融合了：
  - Katyusha 三重动量（Katyusha Momentum）
  - SVRG 方差缩减（Stochastic Variance Reduced Gradient）
  - 概率快照更新（Probabilistic Snapshot Update）

变量定义（每个节点 m 维护）：
  z_m  : (d,) 核心迭代变量（对应 Prompt 算法伪代码中的 z_{m,t}）
  w_m  : (d,) SVRG 锚点变量（快照，以概率 p 更新）
  q_m  : (d,) 动量累积变量
  x_m  : (d,) Katyusha 混合变量（τ·z_m + ρ'·w_m + (1-τ-ρ')·q_m，仅在 step 内使用）
  λ_m  : (d,) 缩放对偶变量（存储于 self.lam_nodes[m]）
  ∇f_m(w_m) : (d,) 在快照 w_m 处的局部全梯度（每次更新 w_m 时重算）

算法伪代码（每个 epoch t = 0, 1, ..., T-1）：

  [Server]：
    y_{t+1} = S_{μ/(ρM)}((1/M)·Σ_m (z_{m,t} + λ_{m,t}))

  [节点 m 并行]：
    x_{m,t+1} = τ·z_{m,t} + ρ'·w_{m,t} + (1-τ-ρ')·q_{m,t}  ← Katyusha 混合
    采样 batch J ⊂ D_m，计算 SVRG 梯度估计：
      v_{m,t+1} = [∇f_{m,J}(x_m) - ∇f_{m,J}(w_m)] + ∇f_m(w_m)
                 （mini-batch 差分 + 全梯度快照）
    z_{m,t+1} = z_{m,t} - (η/γ)·[v_{m,t+1} + ρ·(z_{m,t} - y_{t+1} + λ_{m,t})]
    q_{m,t+1} = x_{m,t+1} + τ·(z_{m,t+1} - z_{m,t})          ← 动量更新
    λ_{m,t+1} = λ_{m,t} + z_{m,t+1} - y_{t+1}
    以概率 p：w_{m,t+1} = q_{m,t+1}；重算 ∇f_m(w_{m,t+1})    ← 快照更新

注意事项：
  1. self.x_nodes[m] 存储 z_m（核心变量），用于指标计算（Gap, Primal Residual）。
     Katyusha 混合变量 x_m 仅在 step() 内部使用，不持久化。
  2. Katyusha 约束：τ + varrho ≤ 1（确保混合系数为凸组合），初始化时断言检查。
  3. Server 使用 z_m 而非 x_m 计算 ū（参考 GGLR 中 pkm_admm 的一致性处理）。
  4. SVRG 梯度估计中，mini-batch 版本：对同一批 J 分别在 x_m 和 w_m 处计算梯度，
     再加上全梯度快照 ∇f_m(w_m) 实现方差缩减。
"""

from typing import List

import numpy as np
import scipy.sparse as sp

from algorithms.base_admm import BaseADMM


class PkmADMM(BaseADMM):
    """
    PKM-ADMM：近端 Katyusha 动量 ADMM（分布式版本）。

    继承 BaseADMM，在基类基础上新增：
      - self.w_nodes      : 各节点 SVRG 锚点变量 w_m
      - self.q_nodes      : 各节点动量变量 q_m
      - self.full_grad_w  : 各节点在 w_m 处的局部全梯度
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
        tau: float = 0.5,
        varrho: float = 0.3,
        update_prob: float = 0.1,
        batch_size: int = 32,
    ) -> None:
        """
        初始化 PKM-ADMM。

        参数（继承自 BaseADMM 的参数见父类 __init__ 文档）
        ------
        eta : float
            基础学习率 η，默认 0.01。
        gamma : float
            步长分母参数 γ，实际步长为 η/γ，默认 1.0。
        tau : float
            Katyusha 动量参数 τ ∈ (0, 1)，默认 0.5。
        varrho : float
            Katyusha 动量参数 ρ'（为避免与 ADMM 惩罚参数 ρ 混淆，命名为 varrho），
            需满足 τ + varrho ≤ 1，默认 0.3。
        update_prob : float
            快照变量 w_m 的随机更新概率 p ∈ (0, 1]，默认 0.1。
            每轮以概率 p 将 w_m 设为 q_m 并重算全梯度快照。
        batch_size : int
            SVRG 估计使用的 mini-batch 大小，默认 32。
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

        # ---- 参数检查 ----
        if tau + varrho > 1.0 + 1e-9:
            raise ValueError(
                f"PKM-ADMM 要求 τ + varrho ≤ 1，"
                f"但当前 τ={tau}, varrho={varrho}, 和={tau + varrho:.4f}。"
            )

        self.eta: float = eta
        self.gamma: float = gamma
        self.tau: float = tau
        self.varrho: float = varrho
        self.update_prob: float = update_prob
        self.batch_size: int = batch_size

        # 有效步长：η/γ
        self._step_size: float = eta / gamma

        # ---- 额外状态变量初始化 ----
        # w_m：SVRG 锚点变量（快照），初始与 z_m 相同（均为零）
        self.w_nodes: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]
        # q_m：Katyusha 动量变量，初始为零
        self.q_nodes: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]
        # ∇f_m(w_m)：在锚点 w_m 处的局部全梯度快照
        # 初始时 w_m = 0，全梯度为 ∇f_m(0) = (1/n_m)·X_m^T·(-b_m·σ(0)) + ν·0
        #                             = (1/n_m)·X_m^T·(-b_m·0.5) = -0.5·(1/n_m)·X_m^T·b_m
        self.full_grad_w: List[np.ndarray] = [
            self.compute_local_grad(m, self.w_nodes[m])
            for m in range(self.M)
        ]

    def _compute_y_update_from_z(self) -> np.ndarray:
        """
        使用 z_m（存储于 self.x_nodes[m]）计算 y 更新。

        PKM-ADMM 的 Server 端聚合使用核心变量 z_m，而非 Katyusha 混合变量 x_m：
            ū_t = (1/M)·Σ_m (z_{m,t} + λ_{m,t})
            y_{t+1} = S_{μ/(ρM)}(ū_t)

        返回
        ----
        y_new : numpy.ndarray, shape (d,)
        """
        # self.x_nodes[m] 存储 z_m
        u_bar: np.ndarray = np.mean(
            [self.x_nodes[m] + self.lam_nodes[m] for m in range(self.M)],
            axis=0,
        )
        threshold: float = self.mu / (self.rho * self.M)
        return self.soft_threshold(u_bar, threshold)

    def step(self, t: int) -> None:
        """
        单步 PKM-ADMM 迭代（t → t+1）。

        执行顺序（符合算法伪代码）：
          1. Server：使用 z_m 聚合并软阈值更新 y_{t+1}
          2. 节点 m：
             a. Katyusha 混合：x_m = τ·z_m + ρ'·w_m + (1-τ-ρ')·q_m
             b. 采样 batch J，计算 SVRG 梯度估计 v_m
             c. 更新 z_m（核心梯度步 + ADMM 惩罚）
             d. 更新 q_m（动量变量）
             e. 更新 λ_m（对偶变量）
             f. 以概率 p 更新 w_m 和全梯度快照

        参数
        ----
        t : int
            当前迭代轮次（0-indexed）。
        """
        # ---- Step 1：Server 使用 z_m 更新 y_{t+1} ----
        self.y = self._compute_y_update_from_z()

        # ---- Step 2：各节点并行更新 ----
        for m in range(self.M):
            # 引用当前节点状态（z_m 存储在 x_nodes[m]）
            z_m: np.ndarray = self.x_nodes[m]          # 当前 z_m = z_{m,t}
            w_m: np.ndarray = self.w_nodes[m]
            q_m: np.ndarray = self.q_nodes[m]
            lam_m: np.ndarray = self.lam_nodes[m]
            n_m: int = self.X_nodes[m].shape[0]
            bs: int = min(self.batch_size, n_m)

            # ---- a. Katyusha 混合变量 x_m（仅在本步内使用）----
            # x_m = τ·z_m + ρ'·w_m + (1-τ-ρ')·q_m
            coeff_q: float = 1.0 - self.tau - self.varrho
            xk_m: np.ndarray = (
                self.tau * z_m
                + self.varrho * w_m
                + coeff_q * q_m
            )

            # ---- b. SVRG 梯度估计 v_{m,t+1} ----
            # 对同一批 J，分别在 x_m 和 w_m 处计算 mini-batch 梯度，取差分：
            #   v_m = ∇f_{m,J}(x_m) - ∇f_{m,J}(w_m) + ∇f_m(w_m)
            # 注意：差分使用相同的批次 J，保证方差缩减的无偏性
            idx: np.ndarray = np.random.choice(n_m, size=bs, replace=False)
            grad_xk: np.ndarray = self.compute_local_grad(m, xk_m, indices=idx)
            grad_wk: np.ndarray = self.compute_local_grad(m, w_m, indices=idx)
            # SVRG 梯度估计：差分 + 全梯度快照（方差缩减核心）
            v_m: np.ndarray = grad_xk - grad_wk + self.full_grad_w[m]

            # ---- c. 更新核心变量 z_m ----
            z_m_old: np.ndarray = z_m.copy()            # 保存 z_{m,t} 用于 q 更新
            # ADMM 惩罚项使用 z_m（核心变量），而非 x_m（混合变量）
            admm_penalty: np.ndarray = self.rho * (
                z_m - self.y + lam_m
            )
            z_m_new: np.ndarray = (
                z_m_old - self._step_size * (v_m + admm_penalty)
            )
            # 更新 x_nodes[m]（存储 z_m）
            self.x_nodes[m] = z_m_new

            # ---- d. 更新动量变量 q_m ----
            # q_{m,t+1} = x_m + τ·(z_{m,t+1} - z_{m,t})
            # 含义：在混合点 x_m 基础上，朝 z 更新方向做 Katyusha 动量加速
            self.q_nodes[m] = xk_m + self.tau * (z_m_new - z_m_old)

            # ---- e. 更新缩放对偶变量 λ_m ----
            # λ_{m,t+1} = λ_{m,t} + z_{m,t+1} - y_{t+1}
            self.lam_nodes[m] = lam_m + z_m_new - self.y

            # ---- f. 以概率 p 更新快照变量 w_m 和全梯度 ----
            # w_m 更新为当前动量变量 q_m，并重算全梯度快照
            if np.random.rand() < self.update_prob:
                self.w_nodes[m] = self.q_nodes[m].copy()
                # 重新计算在新 w_m 处的完整局部梯度（indices=None 即全样本）
                self.full_grad_w[m] = self.compute_local_grad(
                    m, self.w_nodes[m], indices=None
                )
