# -*- coding: utf-8 -*-
"""
algorithms/asvrg_admm.py — ASVRG-ADMM（加速随机方差缩减梯度 ADMM）

算法概述：
  ASVRG-ADMM 融合了加速动量（Nesterov 加速思路）和 SVRG 方差缩减，
  在 ADMM 框架下实现了更快的收敛速率。

  核心创新：引入辅助变量 z_m 和加速插值变量 x_m，
  通过参数 θ 在 z_m（动量点）和快照点 x̃_m 之间插值，
  在保持 SVRG 低方差的同时获得加速效果。

变量定义（每个节点 m 维护）：
  x_tilde[m] : (d,) 外层循环快照点 x̃_m^s（每个外层循环开始时更新）
  full_grad[m]: (d,) 快照点处的完整局部全梯度 ∇f_m(x̃_m^s)（不乘以1/M）
  z_nodes[m] : (d,) 内层循环核心动量变量 z_m（与 x_nodes[m] 分离）
  lam_nodes[m]: (d,) 缩放对偶变量 u_m（与 BaseADMM 对齐）

  注：self.x_nodes[m] 存储"插值变量" x_m = θ·z_m + (1-θ)·x̃_m，
      用于指标计算（Gap, Primal Residual）和 Server 端聚合。

算法伪代码（双层循环）：

  外层循环 s = 0, 1, ..., S-1:
    [节点 m]：
      x_tilde_m = x_nodes_m（上轮末尾的 x_m 作为新快照点）
      full_grad_m = ∇f_m(x_tilde_m)（完整局部全梯度）
    内层循环 t = 0, 1, ..., E-1:
      [Server]：
        y_{t+1} = S_{μ/(ρM)}((1/M)·Σ_m (x_{m,t} + u_{m,t}))
      [节点 m]：
        随机选 i_t ∈ D_m
        SVRG 梯度估计：
          v_m = ∇f_{m,i_t}(x_{m,t}) - ∇f_{m,i_t}(x̃_m) + ∇f_m(x̃_m)
        z 更新（Nesterov 动量步）：
          z_{m,t+1} = z_{m,t} - (η/(γθ))·[(1/M)·v_m + ρ·(z_{m,t} - y_{t+1} + u_{m,t})]
        x 插值更新：
          x_{m,t+1} = θ·z_{m,t+1} + (1-θ)·x̃_m
        对偶更新：
          u_{m,t+1} = u_{m,t} + z_{m,t+1} - y_{t+1}
    快照更新：x̃_m^{s+1} = x_{m,E}

Epoch 计数：
  每次 step(t) 执行一次内层迭代。
  当 t % inner_iter == 0 时，更新快照和全梯度（外层循环起点）。

与 Prompt 公式对应：
  Prompt 中使用 z_m 而非 x_m 进行 Server 聚合：
    y_{t+1} = S_{μ/(ρM)}((1/M)·Σ_m (x_{m,t}^{s+1} + u_{m,t}^{s+1}))
  这里 x_{m,t} 是插值变量（θ·z + (1-θ)·x̃），
  本实现中 self.x_nodes[m] 存储 x_{m,t}（插值变量），
  self.z_nodes[m] 存储 z_{m,t}（核心动量变量）。
"""

from typing import List

import numpy as np
import scipy.sparse as sp

from algorithms.base_admm import BaseADMM


class AsvrgADMM(BaseADMM):
    """
    ASVRG-ADMM：加速 SVRG + 动量插值的分布式 ADMM。

    额外属性：
      self.z_nodes    : List[np.ndarray]，各节点核心动量变量 z_m
      self.x_tilde    : List[np.ndarray]，各节点外层快照点 x̃_m
      self.full_grad  : List[np.ndarray]，各节点快照点处的完整全梯度
      self.inner_iter : int，内层循环长度 E
      self.theta      : float，加速插值系数 θ ∈ (0, 1]
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
        inner_iter: int = 10,
        theta: float = 0.48,
    ) -> None:
        """
        初始化 ASVRG-ADMM。

        参数（继承自 BaseADMM 的参数见父类文档）
        ------
        eta : float
            基础学习率 η（默认 0.01）。
        gamma : float
            步长分母 γ，与 θ 一起控制实际步长 η/(γ·θ)（默认 1.0）。
        batch_size : int
            SVRG 差分估计的 mini-batch 大小（默认 32）。
        inner_iter : int
            内层循环长度 E（默认 10）。每 E 个 epoch 更新一次外层快照。
        theta : float
            加速插值系数 θ ∈ (0, 1]（默认 0.48）。
            θ=1 时退化为标准 SVRG-ADMM；θ<1 时引入加速动量效果。
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
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"ASVRG-ADMM 要求 θ ∈ (0, 1]，当前 theta={theta}。")

        self.eta: float = eta
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.inner_iter: int = inner_iter
        self.theta: float = theta

        # 实际步长：η / (γ·θ)（Nesterov 加速步长）
        self._step_size: float = eta / (gamma * theta)

        # ---- 额外状态变量初始化 ----
        # z_m：核心动量变量，初始为零（与 x_m 初始值相同）
        self.z_nodes: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]
        # x̃_m：外层快照点，初始为零
        self.x_tilde: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]
        # ∇f_m(x̃_m)：快照点处的完整局部全梯度（不乘以 1/M，在 v_m 计算中再除）
        self.full_grad: List[np.ndarray] = [
            self.compute_local_grad(m, self.x_tilde[m])
            for m in range(self.M)
        ]

    def _update_snapshot(self) -> None:
        """
        更新外层快照点和全梯度（在每个外层循环起点调用）。

        x̃_m^{s+1} = x_{m,E}^s（当前 x_m 作为新快照点）
        full_grad_m = ∇f_m(x̃_m^{s+1})（重新计算完整全梯度）

        同时同步 z_m = x̃_m（重置动量变量起点，符合算法初始化约定）
        """
        for m in range(self.M):
            # 快照点更新为当前 x_m（插值变量）
            self.x_tilde[m] = self.x_nodes[m].copy()
            # 动量变量重置为快照点（外层循环开始时 z_{m,0}^{s+1} = x̃_m^{s+1}）
            self.z_nodes[m] = self.x_tilde[m].copy()
            # 重新计算完整全梯度
            self.full_grad[m] = self.compute_local_grad(m, self.x_tilde[m])

    def step(self, t: int) -> None:
        """
        单步 ASVRG-ADMM 迭代（t → t+1）。

        当 t % inner_iter == 0 时，先更新快照（外层循环起点），
        再执行一次内层迭代。

        参数
        ----
        t : int
            当前 epoch 编号（0-indexed）。
        """
        # ---- 外层循环起点：更新快照和全梯度 ----
        if t % self.inner_iter == 0:
            self._update_snapshot()

        # ---- Step 1：Server 使用插值变量 x_m 更新 y ----
        # self.x_nodes[m] 存储 x_{m,t}（插值变量），与 Prompt 公式一致
        self.y = self.compute_y_update()

        # ---- Step 2：各节点并行更新 ----
        for m in range(self.M):
            n_m: int = self.X_nodes[m].shape[0]
            bs: int = min(self.batch_size, n_m)

            # 当前插值变量 x_{m,t} 和动量变量 z_{m,t}
            x_m: np.ndarray = self.x_nodes[m]
            z_m: np.ndarray = self.z_nodes[m]

            # ---- SVRG 梯度估计 ----
            # v_m = ∇f_{m,i_t}(x_{m,t}) - ∇f_{m,i_t}(x̃_m) + ∇f_m(x̃_m)
            # 使用 mini-batch 差分（比单样本更稳定）
            idx: np.ndarray = np.random.choice(n_m, size=bs, replace=False)
            grad_x: np.ndarray = self.compute_local_grad(m, x_m, indices=idx)
            grad_snap: np.ndarray = self.compute_local_grad(
                m, self.x_tilde[m], indices=idx
            )
            # full_grad[m] = ∇f_m(x̃_m)（完整全梯度，不含 1/M）
            v_m: np.ndarray = grad_x - grad_snap + self.full_grad[m]

            # ---- z_m 更新（Nesterov 加速步）----
            # z_{m,t+1} = z_{m,t} - (η/(γθ))·[(1/M)·v_m + ρ·(z_{m,t} - y_{t+1} + u_{m,t})]
            admm_penalty: np.ndarray = self.rho * (
                z_m - self.y + self.lam_nodes[m]
            )
            z_new: np.ndarray = (
                z_m - self._step_size * (v_m / self.M + admm_penalty)
            )
            self.z_nodes[m] = z_new

            # ---- x_m 插值更新 ----
            # x_{m,t+1} = θ·z_{m,t+1} + (1-θ)·x̃_m
            self.x_nodes[m] = (
                self.theta * z_new + (1.0 - self.theta) * self.x_tilde[m]
            )

            # ---- 对偶变量更新（基于 z_{m,t+1}）----
            # u_{m,t+1} = u_{m,t} + z_{m,t+1} - y_{t+1}
            self.lam_nodes[m] = (
                self.lam_nodes[m] + z_new - self.y
            )
