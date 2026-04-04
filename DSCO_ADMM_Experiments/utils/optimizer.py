# -*- coding: utf-8 -*-
"""
utils/optimizer.py — 全局最优值 F* 精确计算

用途：
  预先计算全局目标函数的最优值 F*，供各 ADMM 算法计算 Optimality Gap 使用。

数学推导：
  当共识约束全部满足（x_m = y = x，∀m）时，全局目标退化为：

      F(x) = (1/N)·Σᵢ log(1 + exp(-bᵢ·aᵢᵀ·x)) + (ν/2)·‖x‖₂² + μ·‖x‖₁

  其中 N 为训练样本总数，ν 为 L2 正则系数，μ 为 L1 正则系数。

  ⚠️ 数学一致性保证：上式中：
    - (1/N)·Σᵢ log(1+exp(-bᵢaᵢᵀx)) 对应各节点损失的均值：
        (1/M)·Σ_m (1/n_m)·Σ_{i∈D_m} log(1+exp(-bᵢaᵢᵀx))
      在节点均衡划分（n_m ≈ N/M）时两者严格相等。
    - (ν/2)·‖x‖₂² 与各节点 f_m 中的 L2 正则完全一致。
    - μ·‖x‖₁ 与目标函数中的 μ·‖y‖₁ 完全一致（约束满足时 y=x）。

L1 项处理（变量分裂技巧）：
  由于 L1 项不可微，引入变量分裂：令 x = x⁺ - x⁻（x⁺, x⁻ ≥ 0），
  则 ‖x‖₁ = Σ(x⁺ + x⁻)，目标函数变为关于 (x⁺, x⁻) 的光滑函数（+箱约束），
  可直接用 L-BFGS-B 求解。

数值稳定性：
  - Logistic 损失使用 np.logaddexp(0, -b*score) 避免 exp 溢出。
  - Sigmoid 梯度使用 scipy.special.expit 保证数值稳定。
  - 稀疏矩阵-向量乘法使用 X_train @ x（CSR 格式高效路径）。
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit
from typing import Tuple


# =========================================================================== #
#   公共接口                                                                    #
# =========================================================================== #

def compute_f_star(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    mu: float,
    nu: float,
    x0: np.ndarray = None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    用 L-BFGS-B（变量分裂）精确计算全局最优值 F* 和最优解 x*。

    目标函数（与算法迭代中的 f_m 数学定义严格一致）：
        F(x) = (1/N)·Σᵢ log(1+exp(-bᵢaᵢᵀx)) + (ν/2)‖x‖₂² + μ‖x‖₁

    参数
    ----
    X_train : scipy.sparse.csr_matrix, shape (N, d)
        完整训练集特征矩阵（CSR 稀疏格式）。
    y_train : numpy.ndarray, shape (N,)
        完整训练集标签，值域 {-1, +1}。
    mu : float
        L1 正则系数 μ（稀疏性参数）。
    nu : float
        L2 正则系数 ν（Logistic Regression 权重衰减）。
    x0 : numpy.ndarray, shape (d,), optional
        优化初始点。若为 None，使用零向量。
    tol : float
        L-BFGS-B 梯度收敛容忍度（ftol 和 gtol 均设为此值）。
    max_iter : int
        L-BFGS-B 最大迭代次数（对于 gisette 等大数据集可适当增大）。
    verbose : bool
        是否打印优化过程信息。

    返回
    ----
    F_star : float
        全局最优目标函数值。
    x_star : numpy.ndarray, shape (d,)
        最优解（当约束满足时，算法应收敛至此点附近）。

    注意
    ----
    - gisette 数据集（5000 维）的 L-BFGS-B 优化可能需要数分钟。
    - 若 mu=0，L1 项消失，等价于 L2 正则 Logistic Regression。
    """
    N, d = X_train.shape

    # ---- 准备初始点（2d 维变量：[x⁺; x⁻]，均初始化为 0） ----
    if x0 is not None:
        # 将初始点分解为正负部分
        x0_plus = np.maximum(x0, 0.0)
        x0_minus = np.maximum(-x0, 0.0)
        z0 = np.concatenate([x0_plus, x0_minus])
    else:
        z0 = np.zeros(2 * d, dtype=np.float64)

    # ---- 箱约束：x⁺ ≥ 0，x⁻ ≥ 0 ----
    bounds = [(0.0, None)] * (2 * d)

    # ---- 构造目标函数与梯度的闭包（捕获 X_train, y_train, mu, nu, d, N） ----
    def _obj_and_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算目标函数值及其关于 z = [x⁺; x⁻] 的梯度。

        F(z) = (1/N)·Σ logaddexp(0, -b·(X@x)) + (ν/2)‖x‖² + μ·Σ(x⁺+x⁻)
        其中 x = x⁺ - x⁻

        梯度：
            ∂F/∂x⁺ = (1/N)·X^T·(-b·σ(-b·Xx)) + ν·x + μ·1
            ∂F/∂x⁻ =-(1/N)·X^T·(-b·σ(-b·Xx)) - ν·x + μ·1
        """
        x_plus = z[:d]
        x_minus = z[d:]
        x = x_plus - x_minus  # 原始变量 x

        # --- Logistic 损失及其梯度 ---
        # scores[i] = aᵢᵀx，利用 CSR 稀疏矩阵高效计算
        scores: np.ndarray = X_train @ x                 # shape (N,)

        # logaddexp(0, -b*score) = log(1 + exp(-b·score))，数值稳定
        margins: np.ndarray = -y_train * scores           # shape (N,)
        loss_log: float = float(np.logaddexp(0.0, margins).mean())

        # Sigmoid 梯度系数：coef[i] = -b[i]·σ(-b[i]·score[i])
        # σ(u) = expit(u) 数值稳定
        coefs: np.ndarray = -y_train * expit(margins)    # shape (N,)

        # 梯度：(1/N)·X^T·coef
        # X_train.T @ coefs 可能返回 numpy.matrix，需转为 1D ndarray
        grad_log: np.ndarray = np.asarray(
            X_train.T @ coefs
        ).ravel() / N                                     # shape (d,)

        # --- L2 正则 ---
        loss_l2: float = 0.5 * nu * float(np.dot(x, x))
        grad_l2: np.ndarray = nu * x                     # shape (d,)

        # --- L1 正则（用 x⁺ + x⁻ 表示） ---
        loss_l1: float = mu * (x_plus.sum() + x_minus.sum())
        # ∂/∂x⁺ = μ，∂/∂x⁻ = μ（各维度均为正常数）
        grad_l1_const: np.ndarray = mu * np.ones(d)      # shape (d,)

        # --- 合并目标函数值 ---
        f_val: float = loss_log + loss_l2 + loss_l1

        # --- 合并梯度 ---
        # 对 x⁺：∂F/∂x⁺ = ∂F/∂x · ∂x/∂x⁺ + ∂L1/∂x⁺ = (grad_log+grad_l2) + μ
        # 对 x⁻：∂F/∂x⁻ = ∂F/∂x · ∂x/∂x⁻ + ∂L1/∂x⁻ = -(grad_log+grad_l2) + μ
        grad_x: np.ndarray = grad_log + grad_l2          # shape (d,)
        grad_z: np.ndarray = np.concatenate([
            grad_x + grad_l1_const,   # 对 x⁺ 的梯度
            -grad_x + grad_l1_const,  # 对 x⁻ 的梯度
        ])                                                # shape (2d,)

        return f_val, grad_z

    # ---- 调用 L-BFGS-B 求解 ----
    result = minimize(
        fun=_obj_and_grad,
        x0=z0,
        method="L-BFGS-B",
        jac=True,             # _obj_and_grad 同时返回函数值和梯度
        bounds=bounds,
        options={
            "maxiter": max_iter,
            "ftol":    tol,
            "gtol":    tol,
            "iprint":  1 if verbose else -1,
        },
    )

    if verbose:
        print(f"    [optimizer] L-BFGS-B 状态：{result.message}")
        print(f"    [optimizer] 迭代次数：{result.nit}，"
              f"函数调用次数：{result.nfev}")

    # ---- 还原最优解 x* = x⁺* - x⁻* ----
    z_star: np.ndarray = result.x
    x_star: np.ndarray = z_star[:d] - z_star[d:]
    F_star: float = float(result.fun)

    return F_star, x_star
