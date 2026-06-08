import numpy as np
# import matplotlib.pyplot as plt
# import tenpy
# from tenpy.networks.mps import MPS
# from tenpy.models.tf_ising import TFIChain
# from tenpy.algorithms import dmrg

import scipy.linalg as la


# =======================================================
# 1. MPS 核心代数引擎
# =======================================================
def canonical_compress_mps(mps_list, max_chi):
    """
    使用标准的 DMRG 双向扫描算法 (Two-way Sweeping) 将 MPS 转换为正交形式并截断。
    包含三步：
      1. R-to-L 无截断 QR 扫描 (将正交中心推到最左)
      2. L-to-R SVD 最优纠缠截断 (将正交中心推到最右)
      3. R-to-L 无截断 QR 扫描 (恢复严格的 Right-Orthogonal 形式)

    参数:
        mps_list: 张量列表，每个张量形状为 (L, d_phys, R)
        max_chi: 最大保留的虚拟维度
    返回:
        new_mps: 压缩并重新正交化后的 MPS
    """
    N = len(mps_list)
    new_mps = [t.copy() for t in mps_list]

    # 步骤 1: R-to-L 扫描
    for i in range(N - 1, 0, -1):
        T = new_mps[i]
        L, d_dim, R = T.shape
        M = T.reshape(L, d_dim * R)
        Q, R_mat = la.qr(M.T, mode='economic')
        K = Q.shape[1]
        new_mps[i] = Q.T.reshape(K, d_dim, R)
        left_T = new_mps[i - 1]
        new_mps[i - 1] = np.tensordot(left_T, R_mat.T, axes=([2], [0]))

    # 步骤 2: L-to-R 扫描 (SVD 截断)
    for i in range(N - 1):
        T = new_mps[i]
        L, d_dim, R = T.shape
        M = T.reshape(L * d_dim, R)
        U, S, Vh = la.svd(M, full_matrices=False)
        k = min(len(S), max_chi)
        new_mps[i] = U[:, :k].reshape(L, d_dim, k)
        SVh = np.diag(S[:k]) @ Vh[:k, :]
        right_T = new_mps[i + 1]
        new_mps[i + 1] = np.tensordot(SVh, right_T, axes=([1], [0]))

    # 步骤 3: R-to-L 扫描 (恢复右正交)
    for i in range(N - 1, 0, -1):
        T = new_mps[i]
        L, d_dim, R = T.shape
        M = T.reshape(L, d_dim * R)
        Q, R_mat = la.qr(M.T, mode='economic')
        K = Q.shape[1]
        new_mps[i] = Q.T.reshape(K, d_dim, R)
        left_T = new_mps[i - 1]
        new_mps[i - 1] = np.tensordot(left_T, R_mat.T, axes=([2], [0]))

    # 全局归一化
    T0 = new_mps[0]
    norm0 = np.linalg.norm(T0)
    if norm0 > 1e-12:
        new_mps[0] = T0 / norm0
    return new_mps


def get_orthonormal_complement(V, D):
    """
    求给定等距矩阵 V 的零空间，并将其补全为完整的幺正矩阵。
    """
    m = V.shape[0]
    if m == D: return V
    N_space = la.null_space(V).conj().T
    return np.vstack((V, N_space))


# =======================================================
# 2. 解缠量子线路 (MPD) 构造与演化
# =======================================================
def build_mpo_disentangler(mps, d):
    """
    根据给定的右正交 MPS (\chi=d) 构造单层解缠算符 (MPO 形式)。

    返回:
        mpo: 张量列表，形状统一为 (vL, d_out, d_in, vR)
    """
    N = len(mps)
    mpo = []
    for i, A in enumerate(mps):
        L, d_in, R = A.shape
        # 补齐到 d*d
        pad_A = np.zeros((L, d, d), dtype=complex)
        pad_A[:, :d_in, :R] = A
        V = pad_A.reshape(L, d * d)

        # 补全为满秩幺正矩阵，并取复共轭 (论文要求解缠门)
        U_mat = get_orthonormal_complement(V, d * d)
        W_mat = U_mat.conj()

        # 重组为 (d_out, vL, d_in, vR) -> (vL, d_out, d_in, vR)
        W = W_mat.reshape(d, d, d, d).transpose(1, 0, 2, 3)

        # 处理系统边界的虚拟脚
        if i == 0: W = W[0:1, :, :, :]
        if i == N - 1: W = W[:, :, :, 0:1]
        mpo.append(W)
    return mpo


def apply_mpo_to_mps(mpo, mps):
    """
    将解缠算符 MPO 作用于目标 MPS 上，虚拟维度会随之膨胀。
    """
    new_mps = []
    for W, A in zip(mpo, mps):
        # W: (vL, d_out, d_in, vR), A: (L, d_in, R)
        T = np.tensordot(W, A, axes=([2], [1]))
        # 重排为 ((vL, L), d_out, (vR, R))
        T = T.transpose(0, 3, 1, 2, 4)
        vL, L, d_out, vR, R = T.shape
        T = T.reshape(vL * L, d_out, vR * R)
        new_mps.append(T)
    return new_mps


# =======================================================
# 3. 测量与评估指标
# =======================================================
def get_all_zero_amplitude(mps):
    """高效计算 MPS 与全零初态 |00...0> 的内积幅值"""
    vec = np.eye(1)
    for T in mps:
        M_zero = T[:, 0, :]
        vec = vec @ M_zero
    return np.abs(vec[0, 0])


def mps_overlap(mps1, mps2):
    """计算两个 MPS 的内积 <mps1 | mps2>"""
    N = len(mps1)
    E = np.eye(1)
    for i in range(N):
        A1 = mps1[i].conj()
        A2 = mps2[i]
        tmp = np.tensordot(E, A1, axes=([0], [0]))
        E = np.tensordot(tmp, A2, axes=([0, 1], [0, 1]))
    return np.abs(E[0, 0])


# =======================================================
# 4. 模块独立测试块
# =======================================================
if __name__ == '__main__':
    print("====== 运行 ran2020.py 独立测试 ======")
    N = 6
    d = 2
    chi_init = 4
    np.random.seed(42)

    # 构建随机大维度 MPS
    mps_random = []
    for i in range(N):
        L = 1 if i == 0 else chi_init
        R = 1 if i == N - 1 else chi_init
        mps_random.append(np.random.randn(L, d, R) + 1j * np.random.randn(L, d, R))

    mps_target = canonical_compress_mps(mps_random, chi_init)
    print(f"目标 MPS 成功生成，中间张量形状: {mps_target[N // 2].shape}")

    # 测试提取一层算符并作用
    mps_tilde = canonical_compress_mps(mps_target, d)
    mpo = build_mpo_disentangler(mps_tilde, d)
    mps_next = apply_mpo_to_mps(mpo, mps_target)
    mps_next_compressed = canonical_compress_mps(mps_next, chi_init)

    amp = get_all_zero_amplitude(mps_next_compressed)
    print(f"单层解缠测试完毕，|00..0> 幅值: {amp:.6f}")
    print("==========================================")


