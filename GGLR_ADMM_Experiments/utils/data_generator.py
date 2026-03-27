import numpy as np
import networkx as nx


def generate_graph_incidence_matrix(n_features, p=0.3):
    """生成特征图关联矩阵 D"""
    G = nx.erdos_renyi_graph(n_features, p)
    edges = list(G.edges())
    n_edges = len(edges)

    D = np.zeros((n_edges, n_features))
    for i, (u, v) in enumerate(edges):
        D[i, u] = 1
        D[i, v] = -1
    return D


def generate_gglr_data(n_samples, n_features, D):
    """生成GGLR逻辑回归数据 A, b"""
    A = np.random.randn(n_samples, n_features)
    x_true = np.zeros(n_features)
    x_true[:n_features // 2] = 1.0
    z_true = D @ x_true
    b = np.sign(A @ x_true + 0.1 * np.random.randn(n_samples))
    return A, b, x_true, z_true