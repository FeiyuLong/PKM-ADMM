from config import GLOBAL_SETTINGS, ALGO_PARAMS

from utils.data_generator import generate_graph_incidence_matrix, generate_gglr_data
from utils.plot_utils import plot_convergence_curves_by_time, plot_convergence_curves_by_epoch
from utils.optimizer import compute_gglr_optimal_value

# 导入所有算法
from algorithms.stochastic_admm import stochastic_admm
from algorithms.sag_admm import sag_admm
from algorithms.saga_admm import saga_admm
from algorithms.svrg_admm import svrg_admm
from algorithms.spider_admm import spider_admm
from algorithms.asvrg_admm import asvrg_admm
from algorithms.pkm_admm import pkm_admm

# 算法名称与函数的映射关系
ALGO_MAP = {
    # "STOC-ADMM": stochastic_admm,
    "SAG-ADMM": sag_admm,
    "SAGA-ADMM": saga_admm,
    "SVRG-ADMM": svrg_admm,
    "SPIDER-ADMM": spider_admm,
    "ASVRG-ADMM": asvrg_admm,
    "PKM-ADMM": pkm_admm
}

# 1. 数据准备
n_samples = GLOBAL_SETTINGS["n_samples"]
n_features = GLOBAL_SETTINGS["n_features"]
max_iter = GLOBAL_SETTINGS["max_iter"]

D = generate_graph_incidence_matrix(n_features)
A, b, _, _ = generate_gglr_data(n_samples, n_features, D)

# 2. 计算 p_star
print("=== 开始计算 p_star ===")
p_star = compute_gglr_optimal_value(A, b, D, GLOBAL_SETTINGS["mu1"], GLOBAL_SETTINGS["mu2"])

# 3. 运行算法
results = []
algo_names = list(ALGO_MAP.keys())

for name in algo_names:
    print(f"--- Running {name} ---")

    # 获取该算法专属参数
    params = ALGO_PARAMS[name].copy()

    # 注入公共参数
    params.update({
        "A": A, "b": b, "D": D,
        "max_iter": max_iter,
        "p_star": p_star
    })

    # 调用算法函数并解包字典参数
    res = ALGO_MAP[name](**params)
    results.append(res)

# 4. 绘图
plot_convergence_curves_by_epoch(results, algo_names)
plot_convergence_curves_by_time(results, algo_names)
print("\n=== 实验完成 ===")