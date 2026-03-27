import numpy as np
import time
from utils.metrics import objective_gap, primal_residual, dual_residual

def pkm_admm(A, b, D, max_iter=1000, p_star=0.0,
             mu1 = 1e-3, mu2 = 1e-2, rho = 1.0,
             step_size=0.01, batch_size=32,
             tau=0.5, varrho=0.3, update_prob_p_t=0.1):
    """
    mPG-ADMM求解图诱导正则化逻辑回归 (GGLR) 问题
    目标函数：L(x) + (mu1/2)||x||² + λ||y||₁ s.t. Dx - y = 0
    其中 L(x) = (1/n)∑log(1+exp(-b_i a_i^T x)) 是逻辑回归损失

    参数：
        A: 特征矩阵 (n_samples, n_features)
        b: 标签向量 (n_samples,)
        D: 图关联矩阵 (n_edges, n_features)
        mu1: L2 正则化系数
        mu2: L1 正则化系数 (图引导)
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        step_size: 步长η（对应z-update闭式解中的1/η）
        tau: mPG-ADMM的τ参数（x更新的权重）
        varrho: mPG-ADMM的varrho参数（x更新的权重）
        update_prob_p_t: w更新的概率p_t
        batch_size: 随机批次大小b（对应算法中的J_t大小）
    返回：
        收敛指标（目标间隙、原始残差、对偶残差）
    """
    n, d = A.shape
    n_edges = D.shape[0]

    # ========== 初始化算法变量 ==========
    # 初始点（算法中的x_1, λ_1=0, z_1=w_1=q_1=x_1）
    x = np.zeros(d)
    lam_u = np.zeros(n_edges)  # λ_t（对偶变量）
    z = np.zeros(d)  # z_t
    w = np.zeros(d)  # w_t
    q = np.zeros(d)  # q_t

    # 定义梯度计算函数
    def full_gradient(x):
        """计算全梯度 ∇f(x) = (1/n)A^T(-b/(1+exp(b*A@x))) + mu1*x"""
        logit = A @ x
        sigmoid = 1 / (1 + np.exp(b * logit))
        grad_logreg = (1 / n) * A.T @ (-b * sigmoid)
        return grad_logreg + mu1 * x

    def sample_gradient(x, j):
        """计算单个样本j的梯度 ∇f_j(x) = -b[j]*A[j]/(1+exp(b[j]*A[j]@x))"""
        logit_j = A[j] @ x
        return -b[j] * A[j] / (1 + np.exp(b[j] * logit_j))

    grad_f_w = full_gradient(w)  # 初始全梯度 ∇f(w_1)

    # 收敛指标记录
    gaps, primals, duals = [], [], []
    # 时间指标记录
    time_list = []  # 记录每轮累计时间
    start_time = time.time()  # 算法启动时间

    # ========== 迭代过程 ==========
    for t in range(max_iter):
        # ========== Step 1: y更新（软阈值操作） ==========
        u = D @ z + lam_u  # A=D, B=-I, c=0 → D z_t - y + λ_t = u - y
        y = np.sign(u) * np.maximum(np.abs(u) - mu2 / rho, 0)

        # ========== Step 2: x更新 ==========
        tau_t = tau  # 简化为固定τ_t，可按需调整为动态值
        x_new = tau_t * z + varrho * w + (1 - tau_t - varrho) * q

        # ========== Step 3: 采样mini-batch J_t ==========
        J_t = np.random.choice(n, batch_size, replace=False)

        # ========== Step 4: 计算方差缩减梯度v_{t+1} ==========
        sum_grad_diff = np.zeros(d)
        for j in J_t:
            grad_j_x = sample_gradient(x_new, j)
            grad_j_w = sample_gradient(w, j)
            sum_grad_diff += (grad_j_x - grad_j_w)
        v_t_1 = (1 / batch_size) * sum_grad_diff + grad_f_w
        v_t_1 += mu1 * x_new  # 补充L2正则项的梯度贡献

        # ========== Step 5: z更新（闭式解） ==========
        # 预分解矩阵 (1/η)I + ρD^TD（处理非正定情况）
        pre_matrix = (1 / step_size) * np.eye(d) + rho * D.T @ D
        try:
            L = np.linalg.cholesky(pre_matrix)
        except np.linalg.LinAlgError:
            pre_matrix += 1e-6 * np.eye(d)  # 添加小扰动保证正定
            L = np.linalg.cholesky(pre_matrix)

        # 闭式解计算
        term1 = (1 / step_size) * z - v_t_1
        term2 = rho * D.T @ (y - lam_u)
        z_new = np.linalg.solve(L.T, np.linalg.solve(L, term1 + term2))

        # ========== Step 6: q更新 ==========
        q_new = x_new + tau_t * (z_new - z)

        # ========== Step 7: 对偶变量λ更新 ==========
        lam_u_prev = lam_u.copy()  # 保存更新前值用于计算对偶残差
        lam_u = lam_u + D @ z_new - y  # A=D, B=-I, c=0 → D z - y

        # ========== Step 8: w更新 ==========
        if np.random.rand() < update_prob_p_t:
            w_new = q.copy()  # 用更新前的q_t
            grad_f_w = full_gradient(w_new)  # 重新计算全梯度
        else:
            w_new = w.copy()

        # ========== 更新迭代变量 ==========
        x = x_new
        z = z_new
        q = q_new
        w = w_new

        # ========== 计算收敛指标 ==========
        gap = objective_gap(z, y, D, A, b, mu1, mu2, p_star)
        pr = primal_residual(D, z, y)
        dr = dual_residual(lam_u_prev, lam_u, rho, D)

        gaps.append(gap)
        primals.append(pr)
        duals.append(dr)

        # 计算累计时间并记录
        cum_time = time.time() - start_time
        time_list.append(cum_time)

    return {"gap": gaps, "primal": primals, "dual": duals, "time": time_list}