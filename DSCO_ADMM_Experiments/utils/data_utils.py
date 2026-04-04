# -*- coding: utf-8 -*-
"""
utils/data_utils.py — 数据加载、预处理与节点划分

主要功能：
  1. load_dataset()        — 从磁盘加载 libsvm 格式数据集（支持 .bz2 压缩）
  2. split_data_to_nodes() — 将训练集均匀划分到 M 个分布式节点
  3. compute_node_stats()  — 打印各节点数据分布统计信息（调试用）

数据处理规范：
  - 特征矩阵保持 scipy.sparse.csr_matrix 格式（不转稠密，节省内存）
  - 模型参数 x_m、对偶变量 λ_m 等均为稠密 NumPy 一维数组
  - 标签强制规范化为 {-1, +1}
  - 使用 MaxAbsScaler 将特征缩放至 [-1, 1]（保持稀疏性）
  - 测试集使用训练集的特征维度 d 对齐（避免 w8a 等数据集维度不匹配）

已知数据集特性：
  - a9a:    训练 32561 样本，123 维；a9a.t 测试 16281 样本，123 维
  - w8a:    训练 49749 样本，300 维；w8a.t 测试 14951 样本，300 维
             （测试集实际只有 299 维，必须显式传入 n_features=300）
  - gisette: 训练 6000 样本，5000 维（稀疏压缩 bz2）；
             测试 1000 样本，5000 维
"""

import os
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from typing import List, Tuple, Dict, Optional


# =========================================================================== #
#   公共接口                                                                    #
# =========================================================================== #

def load_dataset(
    train_path: str,
    test_path: Optional[str] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, np.ndarray]:
    """
    加载 libsvm 格式数据集，返回训练集与测试集。

    支持：
      - 标准 libsvm 格式（如 a9a, w8a）
      - bz2 压缩格式（如 gisette_scale.bz2），sklearn 原生支持
      - 若找不到独立测试集，则用 train_test_split 从训练集中划分

    参数
    ----
    train_path : str
        训练集文件路径（可为相对或绝对路径）。
    test_path : str, optional
        测试集文件路径。若为 None 或文件不存在，则从训练集中自动划分。
    test_ratio : float
        当无独立测试集时，从训练集中切分的比例，默认 0.2。
    seed : int
        随机种子，确保划分可复现。

    返回
    ----
    X_train : scipy.sparse.csr_matrix, shape (N_train, d)
        训练集特征矩阵，CSR 稀疏格式。
    y_train : numpy.ndarray, shape (N_train,)
        训练集标签，值域 {-1, +1}。
    X_test : scipy.sparse.csr_matrix, shape (N_test, d)
        测试集特征矩阵，与训练集特征维度对齐。
    y_test : numpy.ndarray, shape (N_test,)
        测试集标签，值域 {-1, +1}。
    """
    # ---- 加载训练集，获取特征维度 d ----
    X_train_raw, y_train_raw = load_svmlight_file(train_path)
    d: int = X_train_raw.shape[1]

    # ---- 判断是否存在独立测试集 ----
    has_test = (test_path is not None) and os.path.exists(test_path)

    if has_test:
        # 必须显式传入 n_features=d，防止测试集特征维度少于训练集
        # （w8a.t 只有 299 维，而训练集有 300 维）
        X_test_raw, y_test_raw = load_svmlight_file(test_path, n_features=d)
    else:
        # 无独立测试集：从训练集中划分
        print(f"    [data_utils] 未找到测试集文件，从训练集中划分 "
              f"{int(test_ratio * 100)}% 作为测试集。")
        (X_train_raw, X_test_raw,
         y_train_raw, y_test_raw) = train_test_split(
            X_train_raw, y_train_raw,
            test_size=test_ratio,
            random_state=seed,
        )

    # ---- 特征缩放：MaxAbsScaler 将特征缩放到 [-1, 1]，保持稀疏性 ----
    scaler = MaxAbsScaler()
    X_train_scaled: sp.csr_matrix = scaler.fit_transform(X_train_raw).tocsr()
    X_test_scaled: sp.csr_matrix = scaler.transform(X_test_raw).tocsr()

    # ---- 标签规范化：强制转换到 {-1, +1} ----
    y_train: np.ndarray = _normalize_labels(y_train_raw)
    y_test: np.ndarray = _normalize_labels(y_test_raw)

    return X_train_scaled, y_train, X_test_scaled, y_test


def split_data_to_nodes(
    X: sp.csr_matrix,
    y: np.ndarray,
    M: int,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[List[sp.csr_matrix], List[np.ndarray]]:
    """
    将训练集均匀划分到 M 个分布式节点。

    采用随机 shuffle 后按 np.array_split 切分，确保各节点样本数之差 ≤ 1。
    每个节点的特征矩阵保持 CSR 稀疏格式，支持高效的稀疏矩阵-向量乘法。

    参数
    ----
    X : scipy.sparse.csr_matrix, shape (N, d)
        完整训练集特征矩阵。
    y : numpy.ndarray, shape (N,)
        完整训练集标签。
    M : int
        节点数量。
    seed : int
        随机种子（用于样本 shuffle）。
    shuffle : bool
        是否在划分前随机打乱样本顺序，默认 True。

    返回
    ----
    X_nodes : list of scipy.sparse.csr_matrix, 长度 M
        X_nodes[m].shape = (n_m, d)，各节点的特征矩阵。
    y_nodes : list of numpy.ndarray, 长度 M
        y_nodes[m].shape = (n_m,)，各节点的标签向量。
    """
    N: int = X.shape[0]

    if shuffle:
        rng = np.random.default_rng(seed)
        perm: np.ndarray = rng.permutation(N)
    else:
        perm = np.arange(N)

    # np.array_split 均匀切分索引数组（最后一组可能多1个样本）
    index_splits: List[np.ndarray] = np.array_split(perm, M)

    X_nodes: List[sp.csr_matrix] = []
    y_nodes: List[np.ndarray] = []

    for idx in index_splits:
        # 切片并确保内存连续的 CSR 格式，利于后续矩阵运算
        X_nodes.append(X[idx].tocsr())
        y_nodes.append(y[idx].copy())

    return X_nodes, y_nodes


def compute_node_stats(
    X_nodes: List[sp.csr_matrix],
    y_nodes: List[np.ndarray],
) -> Dict:
    """
    计算并返回各节点的基本统计信息（调试与日志输出用）。

    参数
    ----
    X_nodes : list of csr_matrix
        各节点特征矩阵列表。
    y_nodes : list of ndarray
        各节点标签列表。

    返回
    ----
    stats : dict
        包含以下键：
          "M"          : int，节点总数
          "d"          : int，特征维度
          "n_per_node" : list[int]，各节点样本数
          "sparsity"   : float，全局平均稀疏度（非零元素比例）
          "pos_ratio"  : list[float]，各节点正样本比例
    """
    M = len(X_nodes)
    d = X_nodes[0].shape[1]
    n_per_node = [X_nodes[m].shape[0] for m in range(M)]

    # 计算全局平均稀疏度
    total_nnz = sum(X_nodes[m].nnz for m in range(M))
    total_elem = sum(n_per_node[m] * d for m in range(M))
    sparsity = total_nnz / total_elem if total_elem > 0 else 0.0

    pos_ratio = [
        float(np.mean(y_nodes[m] > 0)) for m in range(M)
    ]

    stats = {
        "M":          M,
        "d":          d,
        "n_per_node": n_per_node,
        "sparsity":   sparsity,
        "pos_ratio":  pos_ratio,
    }
    return stats


# =========================================================================== #
#   私有辅助函数                                                                #
# =========================================================================== #

def _normalize_labels(y_raw: np.ndarray) -> np.ndarray:
    """
    将任意二分类标签规范化到 {-1, +1}。

    支持输入：
      - 已经是 {-1, +1}：直接返回
      - {0, 1} 格式：将 0 映射为 -1
      - 其他值：将最小值映射为 -1，最大值映射为 +1

    参数
    ----
    y_raw : numpy.ndarray
        原始标签数组。

    返回
    ----
    y : numpy.ndarray, dtype=float64
        规范化后的标签，值域 {-1.0, +1.0}。
    """
    y = y_raw.astype(np.float64).copy()
    unique_vals = np.unique(y)

    if set(unique_vals).issubset({-1.0, 1.0}):
        # 已是 {-1, +1} 格式，无需转换
        return y

    # 将标签映射为 {-1, +1}：最小值→-1，最大值→+1
    min_val, max_val = unique_vals.min(), unique_vals.max()
    y = np.where(y == min_val, -1.0, 1.0)
    return y
