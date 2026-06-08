import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import tenpy
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinChain  # 新增导入
from tenpy.algorithms import dmrg

# 导入底层引擎
import ran2020

# =======================================================
# 1. 核心优化器类 (Ben-Dov 2024)
# =======================================================
class BenDov2024Optimizer:
    def __init__(self, mps_target, num_layers, chi_classical=64, d=2):
        self.mps_target = mps_target
        self.num_layers = num_layers
        self.chi_classical = chi_classical
        self.d = d
        self.N = len(mps_target)

        # 准备全零态 |00...0>
        self.mps_zero = []
        for i in range(self.N):
            T = np.zeros((1, d, 1), dtype=complex)
            T[0, 0, 0] = 1.0
            self.mps_zero.append(T)

        # Layer-by-layer 初始化
        self.mpos = []
        mps_curr = [T.copy() for T in mps_target]
        for l in range(num_layers):
            mps_tilde = ran2020.canonical_compress_mps(mps_curr, max_chi=d)
            mpo = ran2020.build_mpo_disentangler(mps_tilde, d=d)
            self.mpos.append(mpo)
            mps_next = ran2020.apply_mpo_to_mps(mpo, mps_curr)
            mps_curr = ran2020.canonical_compress_mps(mps_next, max_chi=chi_classical)

    def _get_psi_bottom(self, l):
        psi = [T.copy() for T in self.mps_target]
        for m in range(l):
            psi = ran2020.apply_mpo_to_mps(self.mpos[m], psi)
            psi = ran2020.canonical_compress_mps(psi, max_chi=self.chi_classical)
        return psi

    def _get_psi_top(self, l):
        psi = [T.copy() for T in self.mps_zero]
        for m in range(self.num_layers - 1, l, -1):
            W_dagger = [W.transpose(0, 2, 1, 3).conj() for W in self.mpos[m]]
            psi = ran2020.apply_mpo_to_mps(W_dagger, psi)
            psi = ran2020.canonical_compress_mps(psi, max_chi=self.chi_classical)
        return psi

    def optimize_layer(self, l):
        psi_bottom = self._get_psi_bottom(l)
        psi_top = self._get_psi_top(l)

        L_env = [None] * (self.N + 1)
        R_env = [None] * (self.N + 1)
        L_env[0] = np.ones((1, 1, 1), dtype=complex)
        R_env[self.N] = np.ones((1, 1, 1), dtype=complex)

        for i in range(self.N - 1, -1, -1):
            R_env[i] = np.einsum('zrs, xoz, woir, yis -> xwy',
                                 R_env[i+1], psi_top[i].conj(), self.mpos[l][i], psi_bottom[i])

        for i in range(self.N):
            M = np.einsum('xwy, xoz, yis, zrs -> woir',
                          L_env[i], psi_top[i].conj(), psi_bottom[i], R_env[i+1])
            vL_W, d_out, d_in, vR_W = M.shape
            M_trans = M.transpose(1, 0, 2, 3)
            M_mat = M_trans.reshape(d_out * vL_W, d_in * vR_W)

            U, S, Vh = la.svd(M_mat.conj(), full_matrices=False)
            W_opt_mat = U @ Vh

            W_opt_trans = W_opt_mat.reshape(d_out, vL_W, d_in, vR_W)
            W_new = W_opt_trans.transpose(1, 0, 2, 3)
            self.mpos[l][i] = W_new

            L_env[i+1] = np.einsum('xwy, xoz, woir, yis -> zrs',
                                   L_env[i], psi_top[i].conj(), W_new, psi_bottom[i])

    def compute_fidelity(self):
        psi_out = self._get_psi_bottom(self.num_layers)
        overlap = ran2020.mps_overlap(self.mps_zero, psi_out)
        return overlap**2

    def optimize(self, target_iters):
        """执行优化并提取指定迭代次数的保真度"""
        results = {}
        # Iteration 0 对应 Ran 2020 Layer-by-layer 初始值
        results[0] = self.compute_fidelity()

        max_it = max(target_iters)
        for it in range(1, max_it + 1):
            for l in range(self.num_layers):
                self.optimize_layer(l)
            if it in target_iters:
                results[it] = self.compute_fidelity()
        return results

# =======================================================
# 2. 生成目标态函数
# =======================================================
def get_ising_target(N=12, g=1.2, chi_max=64):
    print(f"--- 正在生成 Ising 基态 N={N}, g={g} ---")
    model_params = {'L': N, 'J': 1.0, 'g': g, 'bc_MPS': 'finite', 'conserve': None}
    model = TFIChain(model_params)
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up", "down"] * (N // 2), bc="finite", unit_cell_width=1)
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {'trunc_params': {'chi_max': chi_max}, 'max_trunc_err': 1.0})
    eng.run()

    tensors = []
    for i in range(N):
        B = psi.get_B(i, form='B').to_ndarray()
        if i == 0: B = B.reshape(1, B.shape[1], B.shape[2])
        if i == N - 1: B = B.reshape(B.shape[0], B.shape[1], 1)
        tensors.append(B)
    return tensors


def get_xxz_target(N, Jz, chi_max=64):
    """
    生成 XXZ 模型的基态 MPS (Jx=Jy=1.0)

    Parameters:
        N (int): 系统大小（格点数）
        Jz (float): 各向异性参数
        chi_max (int): DMRG 最大 bond 维度

    Returns:
        list: MPS 张量列表，每个张量形状为 (vL, d, vR)
    """
    model_params = {
        'L': N,
        'S': 0.5,
        'Jx': 1.0,
        'Jy': 1.0,
        'Jz': Jz,
        'bc_MPS': 'finite',
        'conserve': None
    }
    model = SpinChain(model_params)

    # 修正：生成与格点数 N 严格等长的初始状态列表
    # 使用列表推导式，偶数索引为 "up"，奇数索引为 "down"
    initial_state = ["up" if i % 2 == 0 else "down" for i in range(N)]
    psi = MPS.from_product_state(
        model.lat.mps_sites(),
        initial_state,
        bc="finite",
        unit_cell_width=1
    )

    eng = dmrg.TwoSiteDMRGEngine(
        psi, model,
        {'trunc_params': {'chi_max': chi_max}, 'max_trunc_err': 1.0}
    )
    eng.run()

    tensors = []
    for i in range(N):
        B = psi.get_B(i, form='B').to_ndarray()
        if i == 0:
            B = B.reshape(1, B.shape[1], B.shape[2])
        if i == N - 1:
            B = B.reshape(B.shape[0], B.shape[1], 1)
        tensors.append(B)
    return tensors

def compute_direct_encoding_bound(mps_target):
    """计算直接截断编码的保真度下界 (chi0=2 -> L=1, chi0=4 -> L=6)"""
    bounds = {}
    for chi0, L in [(2, 1), (4, 6)]:
        mps_trunc = ran2020.canonical_compress_mps(mps_target, max_chi=chi0)
        overlap = ran2020.mps_overlap(mps_target, mps_trunc)
        bounds[L] = max(1e-12, 1 - overlap**2)
    return bounds

# =======================================================
# 3. XXZ 测试函数
# =======================================================
def test_xxz():
    print("=" * 60)
    print("测试 Ben-Dov 优化器在 XXZ 模型上的表现")
    print("参数范围：Jz 从 0.0 到 2.0")
    print("=" * 60)

    N = 6
    L_fixed = 2  # 或者 L_fixed = 2
    chi_dmrg = 48
    chi_classical = 32  # 经典模拟截断
    target_iters = [128]  # 只需要 128 次迭代的结果

    # Jz 扫描点
    Jz_values = np.linspace(0.0, 2.0, 9)  # 0.0, 0.25, 0.5, ..., 2.0

    print("\nJz\t\tL=0 (init)\tL=128\t\tFidelity (128)")
    print("-" * 60)

    for Jz in Jz_values:
        # 生成目标态
        mps_target = get_xxz_target(N, Jz, chi_dmrg)

        # 创建优化器
        opt = BenDov2024Optimizer(mps_target, num_layers=L_fixed, chi_classical=chi_classical)

        # 获取初始保真度（0次迭代）
        init_fid = opt.compute_fidelity()
        init_inf = max(1e-12, 1 - init_fid)

        # 运行优化直到目标迭代次数
        res = opt.optimize(target_iters)
        final_fid = res[128]
        final_inf = max(1e-12, 1 - final_fid)

        print(f"{Jz:.2f}\t\t{init_inf:.2e}\t\t{final_inf:.2e}\t\t{final_fid:.10f}")

# =======================================================
# 主程序
# =======================================================
if __name__ == '__main__':
    # 原 Fig 2(a) 复现已注释，如需运行取消注释下面一行
    # reproduce_fig2a()

    # 运行 XXZ 测试
    test_xxz()