import numpy as np
from scipy.special import expit

def logistic_loss(x, A, b, mu):
    """逻辑回归损失函数"""
    n = len(b)
    # loss = np.log(1 + np.exp(-b * (A @ x))).mean()
    loss = np.log(1 + np.exp(np.clip(-b * (A @ x), -50, 50))).mean()
    reg = 0.5 * mu * np.linalg.norm(x) ** 2
    return loss + reg

def l1_loss(y, lam):
    """L1正则损失"""
    return lam * np.linalg.norm(y, 1)

def primal_residual(D, x, y):
    """原始残差 ||D@x - y||"""
    return np.linalg.norm(D @ x - y)

def dual_residual(dual_prev, dual_curr, rho, D):
    """对偶残差"""
    return rho * np.linalg.norm(D.T @ (dual_curr - dual_prev))

def objective_gap(x, y, D, A, b, mu, lam, p_star):
    """目标函数间隙"""
    f = logistic_loss(x, A, b, mu)
    h = l1_loss(y, lam)
    return f + h - p_star