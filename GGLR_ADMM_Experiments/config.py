# config.py

# ===================== 1. 全局数据与实验设置 =====================
GLOBAL_SETTINGS = {
    "n_samples": 500,
    "n_features": 50,
    "max_iter": 500,
    "mu1": 1e-3,  # 用于计算 p_star 的参考值
    "mu2": 1e-2,
}

# ===================== 2. 各算法独立超参数 =====================
# A, b, D, max_iter, p_star 会在 main.py 中动态注入，这里只写算法特有参数
ALGO_PARAMS = {
    "STOC-ADMM": {
        "mu1": 1e-3,
        "mu2": 1e-2,
        "rho": 0.5,
        "step_size": 0.001,
        "batch_size": 32
    },
    "SAG-ADMM": {
        "mu1": 1e-3,
        "mu2": 1e-2,
        "rho": 1.0,
        "step_size": 0.01,
        "batch_size": 1
    },
    "SAGA-ADMM": {
        "mu1": 1e-3,
        "mu2": 1e-2,
        "rho": 1.0,
        "step_size": 0.01,
        "batch_size": 32
    },
    "SVRG-ADMM": {
        "mu1": 1e-3,
        "mu2": 1e-2,
        "rho": 1.0,
        "step_size": 0.01,
        "batch_size": 1
    },
    "SPIDER-ADMM": {
        "mu1": 1e-3,
        "mu2": 1e-2,
        "rho": 1.0,
        "step_size": 0.02,
        "batch_size": 1
    },
    "ASVRG-ADMM": {
        "mu1": 1e-3,
        "mu2": 1e-2,
        "rho": 1.0,
        "step_size": 0.01,
        "batch_size": 32,
        "gamma": 1.0,
        "inner_iter": 10,
        "theta": 0.7
    },
    "PKM-ADMM": {
        "mu1": 1e-3,
        "mu2": 1e-2,
        "rho": 1.0,
        "step_size": 0.02,
        "batch_size": 32,
        "gamma": 1.0,
        "tau": 0.5,
        "varrho": 0.3,
        "update_prob": 0.1
    }
}