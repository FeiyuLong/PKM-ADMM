# -*- coding: utf-8 -*-
"""
algorithms/svrg_admm.py — Inexact SVRG-ADMM（随机方差缩减梯度 ADMM）

算法概述：
  SVRG（Stochastic Variance Reduced Gradient）采用周期性全梯度快照策略：
  每隔 K 步（一个内层循环）重新计算一次全梯度快照 ∇f_m(x̃_m)，
  并在内层循环的每步用单样本差分梯度作方差缩减。

  相比 SAG/SAGA 不需要存储 n_m×d 的梯度表，仅需存储快照向量，
  更适合高维数据集（如 gisette 5000 维）。

变量定义（每个节点 m 维护）：
  x_snap[m]    : (d,) 快照点 x̃_m（在外层循环开始时更新）
  full_grad[m] : (d,) 快照点处的全梯度 (1/M)·∇f_m(x̃_m)

算法伪代码（双层循环结构）：

  外层循环 s = 0, 1, ...:
    [节点 m]：快照更新 x̃_m = x_{m,0}^(s)；计算全梯度 (1/M)·∇f_m(x̃_m)
    内层循环 t = 0, 1, ..., K-1:
      [Server]：y_{t+1} = S_{μ/(ρM)}((1/M)·Σ_m (x_{m,t} + λ_{m,t}))
      [节点 m]：
        随机选 j_t，SVRG 梯度估计：
          v_{m,t+1} = (1/M)·(∇f_{m,j_t}(x_{m,t}) - ∇f_{m,j_t}(x̃_m)) + full_grad_m
        x_{m,t+1} = x_{m,t} - (η/γ)·[v_{m,t+1} + ρ·(x_{m,t} - y_{t+1} + λ_{m,t})]
        λ_{m,t+1} = λ_{m,t} + x_{m,t+1} - y_{t+1}
    x_{m,0}^(s+1) = x_{m,K}^(s)  （快照点在下一外层循环起点处更新）

Epoch 计数：
  为与 "1 Epoch = 1 ADMM 迭代" 的定义一致，每次调用 step(t) 执行一次内层迭代。
  当 t 是内层循环的起点（t % K == 0）时，先更新快照和全梯度，再执行迭代。
  这样 Epoch 总数即 max_iter，与其他算法口径统一。

与 Prompt 公式对应：
  v_{m,t+1} = (1/M)·(∇f_{m,j_t}(x_{m,t}) - ∇f_{m,j_t}(x̃_m)) + (1/M)·∇f_m(x̃_m)
  本实现中 full_grad[m] = (1/M)·∇f_m(x̃_m)，与上式一致。
"""

from typing import List

import numpy as np
import scipy.sparse as sp

from algorithms.base_admm import BaseADMM


class SvrgADMM(BaseADMM):
    """
    SVRG-ADMM：基于周期快照全梯度方差缩减的分布式 ADMM。

    额外属性：
      self.x_snap    : List[np.ndarray]，各节点快照点 x̃_m
      self.full_grad : List[np.ndarray]，各节点快照点处的全梯度（已乘 1/M）
      self.update_freq : int，内层循环长度 K（每 K 步更新一次快照）
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
        update_freq: int = 20,
    ) -> None:
        """
        初始化 SVRG-ADMM。

        参数（继承自 BaseADMM 的参数见父类文档）
        ------
        eta : float
            基础学习率 η（默认 0.01）。
        gamma : float
            步长分母 γ，实际步长 = η/γ（默认 1.0）。
        batch_size : int
            SVRG 差分估计的 mini-batch 大小（默认 32）。
            使用相同的 batch J 分别在 x_m 和 x̃_m 处计算梯度。
        update_freq : int
            内层循环长度 K（默认 20）。
            每隔 K 个 epoch 更新一次快照和全梯度。
            较大的 K 减少全梯度计算频率，但快照越旧效果越差；
            较小的 K 快照更新频繁，但每次全梯度计算有额外开销。
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

        # ---- 快照变量初始化 ----
        # 初始快照点：x̃_m = x_{m,0} = 0
        self.x_snap: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]
        # 初始全梯度快照：(1/M)·∇f_m(0)
        # f_m(0) 的梯度 = (1/n_m)·X_m^T·(-b_m·σ(0)) + ν·0
        #               = -0.5·(1/n_m)·X_m^T·b_m
        # 乘以 1/M 后存储
        self.full_grad: List[np.ndarray] = [
            self.compute_local_grad(m, self.x_snap[m]) / self.M
            for m in range(self.M)
        ]

    def _update_snapshot(self) -> None:
        """
        更新所有节点的快照点和全梯度。

        在每个外层循环起点（t % update_freq == 0）调用：
          x̃_m = x_m（当前迭代点）
          full_grad_m = (1/M)·∇f_m(x̃_m)（重新计算完整全梯度）
        """
        for m in range(self.M):
            self.x_snap[m] = self.x_nodes[m].copy()
            # compute_local_grad 返回 ∇f_m(x̃_m)，再乘以 1/M
            self.full_grad[m] = (
                self.compute_local_grad(m, self.x_snap[m]) / self.M
            )

    def step(self, t: int) -> None:
        """
        单步 SVRG-ADMM 迭代（t → t+1）。

        当 t % update_freq == 0 时，先更新快照和全梯度（外层循环起点），
        再执行一次内层 ADMM 迭代。

        参数
        ----
        t : int
            当前 epoch 编号（0-indexed）。
        """
        # ---- 外层循环起点：更新快照 ----
        if t % self.update_freq == 0:
            self._update_snapshot()

        # ---- Step 1：Server 更新 y ----
        self.y = self.compute_y_update()

        # ---- Step 2：各节点并行更新 ----
        for m in range(self.M):
            n_m: int = self.X_nodes[m].shape[0]
            bs: int = min(self.batch_size, n_m)

            # 对同一批 J，分别在 x_m 和 x̃_m 处计算 mini-batch 梯度
            idx: np.ndarray = np.random.choice(n_m, size=bs, replace=False)
            grad_x: np.ndarray = self.compute_local_grad(
                m, self.x_nodes[m], indices=idx
            )
            grad_snap: np.ndarray = self.compute_local_grad(
                m, self.x_snap[m], indices=idx
            )

            # SVRG 梯度估计：差分 + 全梯度快照（方差缩减）
            # v_m = (1/M)·(∇f_{m,J}(x_m) - ∇f_{m,J}(x̃_m)) + (1/M)·∇f_m(x̃_m)
            v_m: np.ndarray = (
                (grad_x - grad_snap) / self.M + self.full_grad[m]
            )

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
