import numpy as np

from utils.data_generator import generate_graph_incidence_matrix, generate_gglr_data
from utils.plot_utils import plot_convergence_curves_by_time, plot_convergence_curves_by_epoch
from utils.optimizer import compute_gglr_optimal_value

# 导入算法
from algorithms.admm import standard_admm
from algorithms.stochastic_admm import stochastic_admm
from algorithms.sag_admm import sag_admm
from algorithms.saga_admm import saga_admm
from algorithms.svrg_admm import svrg_admm
from algorithms.spider_admm import spider_admm
from algorithms.asvrg_admm import asvrg_admm
from algorithms.pkm_admm import pkm_admm


# ===================== 实验参数 =====================
# 默认参数
n_samples = 500 * 1
n_features = 50 * 1
mu1 = 1e-3
mu2 = 1e-2
rho = 1.0
max_iter = 500
step_size = 0.01

# STOC-ADMM 参数
stoc_mu1 = 1e-3              # 默认：1e-3
stoc_mu2 = 1e-2             # 默认：1e-2
stoc_rho = 0.5              # 默认：1.0
stoc_step_size = 0.001       # 默认：0.01
stoc_batch_size = 32        # 默认：32

# SAG-ADMM 参数
sag_mu1 = 1e-3               # 默认：1e-3
sag_mu2 = 1e-2              # 默认：1e-2
sag_rho = 1.0               # 默认：1.0
sag_step_size = 0.01        # 默认：0.01
sag_batch_size = 1              # SAG-ADMM批次大小，默认32

# SAGA-ADMM 参数
saga_mu1 = 1e-3              # 默认：1e-3
saga_mu2 = 1e-2             # 默认：1e-2
saga_rho = 1.0              # 默认：1.0
saga_step_size = 0.01       # 默认：0.01
saga_batch_size = 32             # SAGA-ADMM批次大小，默认32

# SVRG-ADMM 参数
svrg_mu1 = 1e-3              # 默认：1e-3
svrg_mu2 = 1e-2             # 默认：1e-2
svrg_rho = 1.0              # 默认：1.0
svrg_step_size = 0.01       # 默认：0.01
svrg_batch_size = 1             # SVRG-ADMM批次大小，默认32

# SPIDER-ADMM 参数
spider_mu1 = 1e-3            # 默认：1e-3
spider_mu2 = 1e-2           # 默认：1e-2
spider_rho = 1.0            # 默认：1.0
spider_step_size = 0.02     # 默认：0.01
spider_batch_size = 1          # SPIDER-ADMM批次大小，默认32

# ASVRG-ADMM 参数
asvrg_mu1 = 1e-3
asvrg_mu2 = 1e-2
asvrg_rho = 1.0
asvrg_step_size = 0.01
asvrg_batch_size = 32
asvrg_inner_iter = 10     # 全梯度快照更新频率
asvrg_theta = 0.7         # 动量参数
asvrg_gamma = 1.0         # 控制变量

# PKM-ADMM 参数
pkm_mu1 = 1e-3               # 默认：1e-3
pkm_mu2 = 1e-2              # 默认：1e-2
pkm_rho = 1.0               # 默认：1.0
pkm_step_size = 0.02
pkm_batch_size = 32
pkm_gamma = 1.0
pkm_tau = 0.5
pkm_varrho = 0.3
pkm_update_prob = 0.1


# ===================== 生成数据 =====================
D = generate_graph_incidence_matrix(n_features)
A, b, x_true, z_true = generate_gglr_data(n_samples, n_features, D)


# ===================== 计算真实最优值p_star =====================
print("=== 开始用CVXPY(SCS)精确求解GGLR最优值p_star ===")
try:
    p_star = compute_gglr_optimal_value(A, b, D, mu1, mu2)
    print(f"=== CVXPY求解得到p_star = {p_star:.8f} ===")
except RuntimeError as e:
    print(f"=== CVXPY求解失败：{e}，使用p_star=0.0替代 ===")
    p_star = 0.0

# ===================== 运行所有算法 =====================
print("=== 开始运行算法 ===")

results = []

# ======== 比较算法 ========
algo_names = [
    "STOC-ADMM", "SAG-ADMM", "SAGA-ADMM",
    "SVRG-ADMM", "SPIDER-ADMM", "ASVRG-ADMM", "PKM-ADMM"
]

tmp = ' -> '.join(algo_names)
print(f"    按顺序依次比较算法："+tmp)

# 运行算法并检查gap值

# ---- STOC-ADMM ----
print("    Running STOC-ADMM...")
res_stoc  = stochastic_admm(
    A, b, D, max_iter, p_star,
    mu1=stoc_mu1, mu2=stoc_mu2, rho=stoc_rho, step_size=stoc_step_size, batch_size=stoc_batch_size
)
results.append(res_stoc)

# ---- SAG-ADMM ----
print("    Running SAG-ADMM...")
res_sag = sag_admm(
    A, b, D, max_iter=max_iter, p_star=p_star,
    mu1=saga_mu1, mu2=saga_mu2, rho=saga_rho, step_size=saga_step_size, batch_size=sag_batch_size)
results.append(res_sag)

# ---- SAGA-ADMM ----
print("    Running SAGA-ADMM...")
res_saga = saga_admm(
    A, b, D, max_iter, p_star,
    mu1=saga_mu1, mu2=saga_mu2, rho=saga_rho, step_size=sag_step_size
)
results.append(res_saga)

# ---- SVRG-ADMM ----
print("    Running SVRG-ADMM...")
res_svrg = svrg_admm(
    A, b, D, max_iter, p_star,
    mu1=svrg_mu1, mu2=svrg_mu2, rho=svrg_rho, step_size=svrg_step_size, batch_size=svrg_batch_size)
results.append(res_svrg)

# ---- SPIDER-ADMM ----
print("    Running SPIDER-ADMM...")
res_spider = spider_admm(
    A, b, D, max_iter=max_iter, p_star=p_star,
    mu1=spider_mu1, mu2=spider_mu2, rho=spider_rho, step_size=spider_step_size, batch_size=spider_batch_size)
results.append(res_spider)

# ---- ASVRG-ADMM ----
print("    Running ASVRG-ADMM...")
res_asvrg = asvrg_admm(
    A, b, D, max_iter=max_iter, p_star=p_star,
    mu1=asvrg_mu1, mu2=asvrg_mu2, rho=asvrg_rho,
    step_size=asvrg_step_size, batch_size=asvrg_batch_size,
    gamma=asvrg_gamma, inner_iter=asvrg_inner_iter, theta=asvrg_theta
)
results.append(res_asvrg)

# ---- PKM-ADMM ----
print("    Running PKM-ADMM...")
res_pkm = pkm_admm(
    A, b, D,
    max_iter=max_iter,
    p_star=p_star,
    mu1=pkm_mu1,
    mu2=pkm_mu2,
    rho=pkm_rho,
    step_size=pkm_step_size,
    batch_size=pkm_batch_size,
    gamma=pkm_gamma,
    tau=pkm_tau,
    varrho=pkm_varrho,
    update_prob=pkm_update_prob
)
results.append(res_pkm)


# ===================== 绘图并保存 =====================
print("=== 绘制按 Epoch 收敛曲线并保存到 results/ ===")
plot_convergence_curves_by_epoch(results, algo_names)

print("=== 绘制按 Time 收敛曲线并保存到 results/ ===")
plot_convergence_curves_by_time(results, algo_names)

print("=== 实验完成！===")