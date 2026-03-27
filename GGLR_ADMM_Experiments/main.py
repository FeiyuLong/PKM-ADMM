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
from algorithms.pkm_admm import pkm_admm


# ===================== 实验参数 =====================
# 默认参数
n_samples = 500 * 1
n_features = 50 * 1
mu = 1e-3
lam = 1e-2
rho = 1.0
max_iter = 500
step_size = 0.01

# STOC-ADMM 参数
stoc_mu = 1e-3              # 默认：1e-3
stoc_lam = 1e-2             # 默认：1e-2
stoc_rho = 0.5              # 默认：1.0
stoc_step_size = 0.001       # 默认：0.01
stoc_batch_size = 32        # 默认：32

# SAG-ADMM 参数
sag_mu = 1e-3               # 默认：1e-3
sag_lam = 1e-2              # 默认：1e-2
sag_rho = 1.0               # 默认：1.0
sag_step_size = 0.01        # 默认：0.01
sag_batch_size = 1              # SAG-ADMM批次大小，默认32

# SAGA-ADMM 参数
saga_mu = 1e-3              # 默认：1e-3
saga_lam = 1e-2             # 默认：1e-2
saga_rho = 1.0              # 默认：1.0
saga_step_size = 0.01       # 默认：0.01
saga_batch_size = 1             # SAGA-ADMM批次大小，默认32

# SVRG-ADMM 参数
svrg_mu = 1e-3              # 默认：1e-3
svrg_lam = 1e-2             # 默认：1e-2
svrg_rho = 1.0              # 默认：1.0
svrg_step_size = 0.01       # 默认：0.01
svrg_batch_size = 1             # SVRG-ADMM批次大小，默认32

# SPIDER-ADMM 参数
spider_mu = 1e-3            # 默认：1e-3
spider_lam = 1e-2           # 默认：1e-2
spider_rho = 1.0            # 默认：1.0
spider_step_size = 0.02     # 默认：0.01
spider_batch_size = 1          # SPIDER-ADMM批次大小，默认32

# PKM-ADMM 参数
pkm_mu = 1e-3               # 默认：1e-3
pkm_lam = 1e-2              # 默认：1e-2
pkm_rho = 1.0               # 默认：1.0
pkm_step_size = 0.02        # 默认：0.01
pkm_tau = 0.5               # 默认：0.5
pkm_varrho = 0.3            # 默认：0.3
pkm_update_prob = 0.1       # 默认：0.1
pkm_batch_size = 32         # 默认：32


# ===================== 生成数据 =====================
D = generate_graph_incidence_matrix(n_features)
A, b, x_true, z_true = generate_gglr_data(n_samples, n_features, D)


# ===================== 计算真实最优值p_star =====================
print("=== 开始用CVXPY(SCS)精确求解GGLR最优值p_star ===")
try:
    p_star = compute_gglr_optimal_value(A, b, D, mu, lam)
    print(f"=== CVXPY求解得到p_star = {p_star:.8f} ===")
except RuntimeError as e:
    print(f"=== CVXPY求解失败：{e}，使用p_star=0.0替代 ===")
    p_star = 0.0

# ===================== 运行所有算法 =====================
print("=== 开始运行算法 ===")

results = []

# ======== 比较算法 ========
algo_names = [
    "STOC-ADMM", "SAG-ADMM","SAGA-ADMM",
    "SVRG-ADMM", "SPIDER-ADMM", "PKM-ADMM"
]

tmp = ' -> '.join(algo_names)
print(f"    按顺序依次比较算法："+tmp)

# 运行算法并检查gap值

# 1. STOC-ADMM
res_stoc  = stochastic_admm(
    A, b, D, max_iter, p_star,
    mu=stoc_mu, lam=stoc_lam, rho=stoc_rho, step_size=stoc_step_size, batch_size=stoc_batch_size
)
results.append(res_stoc)

# 2. SAG-ADMM
res_sag = sag_admm(
    A, b, D, mu=saga_mu, lam=saga_lam, rho=saga_rho, max_iter=max_iter, step_size=saga_step_size, p_star=p_star, batch_size=sag_batch_size)
# print("SAG-ADMM gap 前10个值:", res_sag["gap"][:10])
# print("    SAG-ADMM gap 是否有inf:", np.any(np.isinf(res_sag["gap"])))
# print("    SAG-ADMM gap 是否有NaN:", np.any(np.isnan(res_sag["gap"])))
results.append(res_sag)

res_saga = saga_admm(
    A, b, D, max_iter, p_star,
    mu=saga_mu, lam=saga_lam, rho=saga_rho, step_size=sag_step_size
)
results.append(res_saga)

# 3. SVRG-ADMM
res_svrg = svrg_admm(
    A, b, D, max_iter, p_star,
    mu=svrg_mu, lam=svrg_lam, rho=svrg_rho, step_size=svrg_step_size, batch_size=svrg_batch_size)
# print("SVRG-ADMM gap 前10个值:", res_svrg["gap"][:10])
# print("    SVRG-ADMM gap 是否有inf:", np.any(np.isinf(res_svrg["gap"])))
# print("    SVRG-ADMM gap 是否有NaN:", np.any(np.isnan(res_svrg["gap"])))
results.append(res_svrg)

# 4. SPIDER-ADMM
res_spider = spider_admm(
    A, b, D, mu=spider_mu, lam=spider_lam, rho=spider_rho, max_iter=max_iter, step_size=spider_step_size, p_star=p_star, batch_size=spider_batch_size)
# print("SPIDER-ADMM gap 前10个值:", res_spider["gap"][:10])
# print("    SPIDER-ADMM gap 是否有inf:", np.any(np.isinf(res_spider["gap"])))
# print("    SPIDER-ADMM gap 是否有NaN:", np.any(np.isnan(res_spider["gap"])))
results.append(res_spider)

# 5. PKM-ADMM
res_pkm = pkm_admm(
    A, b, D,
    mu=pkm_mu,
    lam=pkm_lam,
    rho=pkm_rho,
    max_iter=max_iter,
    step_size=pkm_step_size,
    tau=pkm_tau,
    varrho=pkm_varrho,
    update_prob_p_t=pkm_update_prob,
    batch_size=pkm_batch_size,
    p_star=p_star
)
# print("PKM-ADMM gap 前10个值:", res_pkm["gap"][:10])
# print("    PKM-ADMM gap 是否有inf:", np.any(np.isinf(res_pkm["gap"])))
# print("    PKM-ADMM gap 是否有NaN:", np.any(np.isnan(res_pkm["gap"])))
results.append(res_pkm)


# ===================== 绘图并保存 =====================
print("=== 绘制按 Epoch 收敛曲线并保存到 results/ ===")
plot_convergence_curves_by_epoch(results, algo_names)

print("=== 绘制按 Time 收敛曲线并保存到 results/ ===")
plot_convergence_curves_by_time(results, algo_names)

print("=== 实验完成！===")