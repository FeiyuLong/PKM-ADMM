# -*- coding: utf-8 -*-
"""
algorithms/base_admm.py — 分布式 ADMM 抽象基类

定义所有 DSCO-ADMM 变体共用的数据结构、工具方法和主循环逻辑。
具体算法（STOC-ADMM、PKM-ADMM 等）继承本类，仅需实现 step() 方法。

问题形式：
    min_{x_m, y}  (1/M)·Σ_m f_m(x_m)  +  μ·‖y‖₁
    s.t.  x_m = y = 0, ∀m ∈ [M]

    其中：f_m(x_m) = (1/n_m)·Σ_{i∈D_m} log(1+exp(-bᵢ·aᵢᵀ·x_m)) + (ν/2)·‖x_m‖²

ADMM 增广拉格朗日法（使用缩放对偶变量 λ_m = u_m/ρ）：
    y_{t+1}   = S_{μ/(ρM)}( (1/M)·Σ_m (x_{m,t} + λ_{m,t}) )
    x_{m,t+1} = argmin_{x_m} { f_m(x_m) + (ρ/2)·‖x_m - y_{t+1} + λ_{m,t}‖² }
                ≈ 子类实现近似/随机更新
    λ_{m,t+1} = λ_{m,t} + x_{m,t+1} - y_{t+1}

Epoch 定义：1 Epoch = 1 次完整的 ADMM 迭代（t → t+1），
每 epoch 结束后立即记录所有评价指标和 Wall-clock 时间。
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.special import expit


# =========================================================================== #
#   抽象基类                                                                    #
# =========================================================================== #

class BaseADMM(ABC):
    """
    分布式稀疏共识优化（DSCO）ADMM 的抽象基类。

    子类约定：
      - 重写 step(t: int) 方法，完成一次完整的 ADMM 迭代。
      - step() 执行后，self.x_nodes、self.y、self.lam_nodes 均更新为 t+1 时刻的值。
      - 若算法额外维护辅助变量（如 w_nodes, q_nodes），在子类 __init__ 中初始化。

    属性（子类可直接访问）：
      self.M           : int，节点数量
      self.d           : int，特征维度
      self.X_nodes     : List[csr_matrix]，各节点训练特征
      self.y_nodes     : List[ndarray]，各节点训练标签
      self.X_test      : csr_matrix，测试集特征
      self.y_test      : ndarray，测试集标签
      self.mu          : float，L1 正则系数
      self.nu          : float，L2 正则系数
      self.rho         : float，ADMM 惩罚参数
      self.F_star      : float，预计算的全局最优目标值
      self.max_iter    : int，最大迭代轮次
      self.x_nodes     : List[ndarray (d,)]，各节点局部变量 x_m
      self.lam_nodes   : List[ndarray (d,)]，各节点缩放对偶变量 λ_m
      self.y           : ndarray (d,)，全局共识变量
      self.history     : Dict[str, List[float]]，收敛指标历史
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
    ) -> None:
        """
        初始化基类公共状态。

        参数
        ----
        X_nodes : list of csr_matrix
            各节点的训练特征矩阵，X_nodes[m].shape = (n_m, d)。
        y_nodes : list of ndarray
            各节点的训练标签，y_nodes[m].shape = (n_m,)，值域 {-1, +1}。
        X_test : csr_matrix, shape (N_test, d)
            测试集特征矩阵，用于计算分类准确率。
        y_test : ndarray, shape (N_test,)
            测试集标签，值域 {-1, +1}。
        mu : float
            L1 正则系数 μ（对应目标函数中的 μ·‖y‖₁ 项）。
        nu : float
            L2 正则系数 ν（对应局部损失中的 (ν/2)·‖x_m‖² 项）。
        rho : float
            ADMM 惩罚参数 ρ（增广拉格朗日惩罚系数）。
        F_star : float
            全局最优目标函数值 F*，由 utils.optimizer.compute_f_star() 预计算。
        max_iter : int
            最大迭代轮次（即最大 Epoch 数）。
        seed : int
            随机种子，确保实验可复现（子类中随机采样使用同一种子）。
        """
        # ---- 基本维度 ----
        self.M: int = len(X_nodes)
        self.d: int = X_nodes[0].shape[1]

        # ---- 数据 ----
        self.X_nodes: List[sp.csr_matrix] = X_nodes
        self.y_nodes: List[np.ndarray] = y_nodes
        self.X_test: sp.csr_matrix = X_test
        self.y_test: np.ndarray = y_test

        # ---- 超参数 ----
        self.mu: float = mu
        self.nu: float = nu
        self.rho: float = rho
        self.F_star: float = F_star
        self.max_iter: int = max_iter

        # ---- 随机种子 ----
        np.random.seed(seed)

        # ---- 优化变量初始化（全零起点）----
        # x_nodes[m]：节点 m 的局部主变量，shape (d,)，稠密
        self.x_nodes: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]
        # lam_nodes[m]：节点 m 的缩放对偶变量 λ_m = u_m/ρ，shape (d,)，稠密
        self.lam_nodes: List[np.ndarray] = [
            np.zeros(self.d, dtype=np.float64) for _ in range(self.M)
        ]
        # y：全局共识变量（Server 端维护），shape (d,)，稠密
        self.y: np.ndarray = np.zeros(self.d, dtype=np.float64)

        # ---- 收敛历史记录（每 epoch 追加一个值）----
        self.history: Dict[str, List[float]] = {
            "gap":      [],   # Optimality Gap：|F(x_t, y_t) - F*|
            "primal":   [],   # Primal Residual：√((1/M)·Σ_m ‖x_m - y‖²)
            "dual":     [],   # Dual Residual：ρ·‖y_t - y_{t-1}‖
            "accuracy": [],   # Test Accuracy：全局测试集准确率
            "time":     [],   # 累计 Wall-clock 时间（秒）
        }

        # ---- 内部计时与状态 ----
        self._start_time: float = 0.0
        self._y_prev: np.ndarray = np.zeros(self.d, dtype=np.float64)

    # ======================================================================= #
    #   工具方法（子类可直接调用）                                              #
    # ======================================================================= #

    def soft_threshold(
        self,
        u: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        软阈值算子（Soft-Thresholding / Proximal Operator of L1）：

            S_λ(u) = sign(u) · max(|u| - λ, 0)

        这是 L1 正则项 λ·‖·‖₁ 的近端算子闭合解。
        在 y 更新步骤中，threshold = μ / (ρ·M)。

        参数
        ----
        u : numpy.ndarray，任意形状
            输入向量（或矩阵）。
        threshold : float
            阈值 λ ≥ 0。

        返回
        ----
        numpy.ndarray，与 u 同形状。
        """
        return np.sign(u) * np.maximum(np.abs(u) - threshold, 0.0)

    def compute_y_update(self) -> np.ndarray:
        """
        Server 端执行 y 更新步骤：

            ū_t = (1/M)·Σ_m (x_{m,t} + λ_{m,t})
            y_{t+1} = S_{μ/(ρ·M)}(ū_t)

        注：λ_m 为缩放对偶变量（已除以 ρ），所以聚合时不再乘以 ρ。

        返回
        ----
        y_new : numpy.ndarray, shape (d,)
            更新后的全局共识变量。
        """
        # 计算各节点 (x_m + λ_m) 的均值
        u_bar: np.ndarray = np.mean(
            [self.x_nodes[m] + self.lam_nodes[m] for m in range(self.M)],
            axis=0,
        )
        # 软阈值算子，阈值 = μ / (ρ·M)
        threshold: float = self.mu / (self.rho * self.M)
        return self.soft_threshold(u_bar, threshold)

    def compute_local_grad(
        self,
        m: int,
        x_m: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        计算节点 m 在 x_m 处的局部梯度（完整梯度或 mini-batch 梯度）。

        局部损失：f_m(x_m) = (1/n_m)·Σ_{i∈D_m} log(1+exp(-bᵢaᵢᵀx_m)) + (ν/2)‖x_m‖²

        完整梯度（indices=None）：
            ∇f_m(x_m) = (1/n_m)·X_m^T·(-b_m·σ(-b_m·X_m·x_m)) + ν·x_m

        Mini-batch 梯度（indices 指定批次下标）：
            g_m = (1/|I|)·X_I^T·(-b_I·σ(-b_I·X_I·x_m)) + ν·x_m

        参数
        ----
        m : int
            节点编号（0-indexed）。
        x_m : numpy.ndarray, shape (d,)
            当前节点局部变量。
        indices : numpy.ndarray, optional
            批次样本下标（整数数组）。若为 None，使用全部样本。

        返回
        ----
        grad : numpy.ndarray, shape (d,)
            局部梯度（稠密数组）。

        数值稳定性：
          - X_m @ x_m：稀疏矩阵-稠密向量乘法，高效且精确。
          - expit(-b_m * scores)：等价于 σ(b_m * scores)，数值稳定。
          - X_m.T @ coefs 可能返回 numpy.matrix，强制转为 1D ndarray。
        """
        X_m: sp.csr_matrix = self.X_nodes[m]
        b_m: np.ndarray = self.y_nodes[m]

        # 根据 indices 选取 mini-batch 或全样本
        if indices is not None:
            X_m = X_m[indices]
            b_m = b_m[indices]
        n_batch: int = X_m.shape[0]

        # 计算样本得分：score[i] = aᵢᵀ·x_m
        scores: np.ndarray = X_m @ x_m                       # shape (n_batch,)

        # 数值稳定的 Logistic 梯度系数：coef[i] = -b[i]·σ(-b[i]·score[i])
        # σ(u) = expit(u) = 1/(1+exp(-u))
        coefs: np.ndarray = -b_m * expit(-b_m * scores)      # shape (n_batch,)

        # 梯度的 Logistic 部分：(1/n_batch)·X_m^T·coef
        # X_m.T @ coefs 对于 CSR 格式会自动转为 CSC 加速，结果可能是 matrix
        grad_log: np.ndarray = np.asarray(
            X_m.T @ coefs
        ).ravel() / n_batch                                   # shape (d,)

        # L2 正则梯度：ν·x_m
        grad_l2: np.ndarray = self.nu * x_m                  # shape (d,)

        return grad_log + grad_l2

    # ======================================================================= #
    #   指标记录                                                               #
    # ======================================================================= #

    def _compute_metrics(self) -> None:
        """
        计算当前迭代的所有评价指标，并追加到 self.history。

        该方法在每个 epoch 的 step() 调用之后立即执行，确保记录频率为
        1 Epoch = 1 ADMM 迭代。

        指标定义：
          Gap      = |F(x_m, y) - F*|
                     = |(1/M)·Σ_m f_m(x_m) + μ·‖y‖₁ - F*|
          Primal   = √((1/M)·Σ_m ‖x_m - y‖²)
          Dual     = ρ·‖y_t - y_{t-1}‖              （y_prev 由 run() 传入）
          Accuracy = mean(sign(X_test @ y) == y_test)
          Time     = time.time() - self._start_time  （累计秒数）
        """
        # ---- 1. Optimality Gap ----
        obj_val: float = 0.0
        for m in range(self.M):
            X_m = self.X_nodes[m]
            b_m = self.y_nodes[m]
            x_m = self.x_nodes[m]
            n_m: int = X_m.shape[0]

            scores: np.ndarray = X_m @ x_m                   # (n_m,)
            # logaddexp(0, -b*score) = log(1+exp(-b*score))，数值稳定
            loss_m: float = float(
                np.logaddexp(0.0, -b_m * scores).mean()
            )
            l2_m: float = 0.5 * self.nu * float(np.dot(x_m, x_m))
            obj_val += (loss_m + l2_m) / self.M

        obj_val += self.mu * float(np.linalg.norm(self.y, ord=1))
        gap: float = abs(obj_val - self.F_star)

        # ---- 2. Primal Residual ----
        # R^p = √((1/M)·Σ_m ‖x_m - y‖²)
        primal: float = float(np.sqrt(
            np.mean([
                np.dot(self.x_nodes[m] - self.y, self.x_nodes[m] - self.y)
                for m in range(self.M)
            ])
        ))

        # ---- 3. Dual Residual ----
        # R^d = ρ·‖y_t - y_{t-1}‖，y_prev 由 run() 在 step() 前保存
        dual: float = float(
            self.rho * np.linalg.norm(self.y - self._y_prev)
        )

        # ---- 4. Test Accuracy ----
        # 预测：sign(X_test @ y)
        test_scores: np.ndarray = self.X_test @ self.y       # (N_test,)
        if sp.issparse(test_scores):
            test_scores = np.asarray(test_scores).ravel()
        pred: np.ndarray = np.sign(test_scores)
        pred[pred == 0] = 1.0                                 # 边界情况：得分为 0 → +1
        accuracy: float = float(np.mean(pred == self.y_test))

        # ---- 5. Wall-clock 时间 ----
        elapsed: float = time.time() - self._start_time

        # ---- 追加到历史记录 ----
        self.history["gap"].append(gap)
        self.history["primal"].append(primal)
        self.history["dual"].append(dual)
        self.history["accuracy"].append(accuracy)
        self.history["time"].append(elapsed)

    # ======================================================================= #
    #   主循环                                                                  #
    # ======================================================================= #

    def run(self) -> Dict[str, List[float]]:
        """
        执行 ADMM 主循环，每 epoch 调用子类 step(t) 并立即记录指标。

        循环结构（1 Epoch = 1 ADMM 迭代）：

            For t = 0, 1, ..., max_iter-1:
                1. 保存 y_prev（用于计算对偶残差）
                2. 调用 self.step(t)       ← 子类实现
                3. 调用 self._compute_metrics()

        返回
        ----
        history : dict
            包含 "gap", "primal", "dual", "accuracy", "time" 的列表，
            每个列表长度为 max_iter。
        """
        self._start_time = time.time()
        self._y_prev = self.y.copy()

        for t in range(self.max_iter):
            # 保存当前 y，用于对偶残差计算
            self._y_prev = self.y.copy()

            # 执行一步 ADMM 迭代（子类具体实现）
            self.step(t)

            # 记录本 epoch 的所有评价指标
            self._compute_metrics()

        return self.history

    # ======================================================================= #
    #   抽象方法（子类必须实现）                                               #
    # ======================================================================= #

    @abstractmethod
    def step(self, t: int) -> None:
        """
        单步 ADMM 迭代（t → t+1），子类必须重写此方法。

        执行完毕后，以下属性应更新为 t+1 时刻的值：
          - self.x_nodes[m]   : 各节点局部变量
          - self.y            : 全局共识变量
          - self.lam_nodes[m] : 各节点缩放对偶变量

        参数
        ----
        t : int
            当前迭代轮次（0-indexed），范围 [0, max_iter)。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 step() 方法。"
        )
