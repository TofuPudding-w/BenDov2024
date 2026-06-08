"""
Reproduce Ben-Dov et al. (2024) "Approximate encoding of quantum states using shallow circuits"

复现论文核心结果：
  Fig 2(a): Ising 模型 — 保真度随迭代次数变化（不同 N 和 L）
  Fig 3:    XXZ 模型 — 保真度随 Jz 变化
  Comparison with direct truncation bounds (Ran 2020)

使用:
  python BenDov2024/main.py          # 运行全部复现
  python BenDov2024/main.py --quick  # 快速测试（小参数）
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# 确保能导入 BenDov2024 包
sys.path.insert(0, os.path.dirname(__file__))
from BenDov2024 import BenDov2024Optimizer, get_ising_target, get_xxz_target, compute_direct_encoding_bound
import ran2020

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =======================================================
# Fig 2(a): Ising model — Infidelity vs Iterations
# =======================================================
def reproduce_fig2a(N_list=None, L_list=None, g=1.2, chi_dmrg=64, chi_classical=64,
                    target_iters=None, quick=False):
    """复现 Fig 2(a): Ising 基态制备保真度

    参数:
        N_list: 系统大小列表
        L_list: 层数列表（每层 = 1 个解缠门）
        g: Ising 横向场
        chi_dmrg: DMRG 截断
        chi_classical: 经典模拟截断
        target_iters: 要记录的迭代次数列表（默认 2 的幂次）
        quick: 快速模式（更小的参数）
    """
    if quick:
        N_list = N_list or [6, 8]
        L_list = L_list or [1, 2, 3]
        chi_dmrg = 32
        chi_classical = 32
        target_iters = target_iters or [1, 2, 4, 8, 16, 32]
    else:
        N_list = N_list or [6, 8, 10, 12]
        L_list = L_list or [1, 2, 3, 4, 5, 6]
        chi_dmrg = 64
        chi_classical = 64
        target_iters = target_iters or [2**k for k in range(8)]  # 1, 2, 4, ..., 128

    print("=" * 60)
    print(f"Reproducing Fig 2(a): Ising model, g={g}")
    print(f"N ∈ {N_list}, L ∈ {L_list}")
    print(f"Target iterations: {target_iters}")
    print("=" * 60)

    # 存储所有结果: results[N][L] = {iter: infidelity}
    all_results = {}

    for N in N_list:
        print(f"\n{'─'*50}")
        print(f"N = {N}")
        print(f"{'─'*50}")

        # 生成 Ising 基态
        t0 = time.time()
        mps_target = get_ising_target(N, g=g, chi_max=chi_dmrg)
        dt_dmrg = time.time() - t0
        print(f"  DMRG complete ({dt_dmrg:.1f}s)")

        # 计算直接截断边界（L=1 对应 chi0=2, L=6 对应 chi0=4）
        direct_bounds = compute_direct_encoding_bound(mps_target)

        all_results[N] = {'direct_bounds': direct_bounds, 'layers': {}}

        for L in L_list:
            print(f"  L = {L} layers...", end=' ', flush=True)
            t0 = time.time()

            opt = BenDov2024Optimizer(mps_target, num_layers=L,
                                       chi_classical=chi_classical)
            init_fid = opt.compute_fidelity()
            results = opt.optimize(target_iters)

            dt = time.time() - t0
            print(f"done ({dt:.1f}s)")

            # 存储：迭代 0 = 初始，迭代 i = 第 i 轮后的保真度
            all_results[N]['layers'][L] = {
                0: max(1e-12, 1 - init_fid)  # infidelity
            }
            for it, fid in results.items():
                all_results[N]['layers'][L][it] = max(1e-12, 1 - fid)

    # ── 绘图 ──
    fig, axes = plt.subplots(1, len(N_list), figsize=(5 * len(N_list), 5),
                              squeeze=False)
    axes = axes[0]

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(L_list)))
    markers = ['o', 's', 'D', '^', 'v', '<']

    for idx, N in enumerate(N_list):
        ax = axes[idx]
        data = all_results[N]

        # 直接截断边界
        for L_bound, infid in data['direct_bounds'].items():
            ax.axhline(y=infid, color='gray', linestyle='--', alpha=0.5,
                        label=f'Direct trunc (L={L_bound})' if idx == 0 else '')

        # 逐层优化结果
        for li, L in enumerate(L_list):
            if L not in data['layers']:
                continue
            layer_data = data['layers'][L]
            iters = sorted(layer_data.keys())
            infids = [layer_data[i] for i in iters]

            ax.loglog(iters, infids, marker=markers[li % len(markers)],
                       color=colors[li], linewidth=1.5, markersize=5,
                       label=f'L={L}', alpha=0.85)

        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Infidelity (1 − F)', fontsize=12)
        ax.set_title(f'N = {N}', fontsize=14)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0.8, max(target_iters) * 1.5)
        ax.set_ylim(1e-6, 1.5)

    fig.suptitle('Fig 2(a): Ising Model — Infidelity vs Iterations\n'
                 f'g = {g}, χ_dmrg = {chi_dmrg}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2a_ising.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")

    return all_results


# =======================================================
# Fig 3: XXZ model — Infidelity vs Jz
# =======================================================
def reproduce_fig_xxz(N=6, L=2, Jz_values=None, chi_dmrg=48, chi_classical=32,
                       target_iter=128, quick=False):
    """复现 XXZ 模型结果

    参数:
        N: 系统大小
        L: 层数
        Jz_values: Jz 扫描点
        target_iter: 目标迭代次数
    """
    if quick:
        Jz_values = Jz_values or np.linspace(0.0, 2.0, 5)
        chi_dmrg = 32
        target_iter = 32
    else:
        Jz_values = Jz_values or np.linspace(0.0, 2.0, 9)

    print(f"\n{'='*60}")
    print(f"Reproducing XXZ results: N={N}, L={L}")
    print(f"Jz ∈ [{Jz_values[0]:.1f}, {Jz_values[-1]:.1f}]")
    print(f"Target iteration: {target_iter}")
    print("=" * 60)

    Jz_list = []
    init_infids = []
    final_infids = []

    for Jz in Jz_values:
        print(f"  Jz = {Jz:.2f}...", end=' ', flush=True)
        t0 = time.time()

        mps_target = get_xxz_target(N, Jz, chi_dmrg)
        opt = BenDov2024Optimizer(mps_target, num_layers=L,
                                   chi_classical=chi_classical)

        init_fid = opt.compute_fidelity()
        init_infid = max(1e-12, 1 - init_fid)

        res = opt.optimize([target_iter])
        final_fid = res[target_iter]
        final_infid = max(1e-12, 1 - final_fid)

        dt = time.time() - t0
        print(f"done ({dt:.1f}s) — infidelity: {init_infid:.2e} → {final_infid:.2e}")

        Jz_list.append(Jz)
        init_infids.append(init_infid)
        final_infids.append(final_infid)

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(Jz_list, init_infids, 's--', color='#1f77b4', linewidth=1.5,
                markersize=8, label=f'Initial (0 iters)')
    ax.semilogy(Jz_list, final_infids, 'o-', color='#d62728', linewidth=2,
                markersize=8, label=f'Optimized ({target_iter} iters)')

    ax.set_xlabel('Jz (anisotropy)', fontsize=12)
    ax.set_ylabel('Infidelity (1 − F)', fontsize=12)
    ax.set_title(f'XXZ Model: N = {N}, L = {L}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_xxz.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    return Jz_list, init_infids, final_infids


# =======================================================
# Layer scaling: Infidelity vs number of layers (fixed N)
# =======================================================
def reproduce_layer_scaling(N=10, L_max=6, g=1.2, chi_dmrg=64, chi_classical=64,
                             target_iters=None, quick=False):
    """复现：固定 N，层数对保真度的影响"""
    if quick:
        N = 6
        L_max = 4
        chi_dmrg = 32
        chi_classical = 32
        target_iters = target_iters or [1, 4, 16, 32]
    else:
        target_iters = target_iters or [2**k for k in range(8)]

    print(f"\n{'='*60}")
    print(f"Layer scaling: N={N}, L ∈ [1, {L_max}]")
    print("=" * 60)

    mps_target = get_ising_target(N, g=g, chi_max=chi_dmrg)
    direct_bounds = compute_direct_encoding_bound(mps_target)

    L_values = list(range(1, L_max + 1))
    final_infids = {it: [] for it in target_iters if it > 0}

    for L in L_values:
        print(f"  L = {L}...", end=' ', flush=True)
        opt = BenDov2024Optimizer(mps_target, num_layers=L,
                                   chi_classical=chi_classical)
        results = opt.optimize(target_iters)
        for it in final_infids:
            final_infids[it].append(max(1e-12, 1 - results[it]))
        print("done")

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(final_infids)))

    for ci, (it, infids) in enumerate(final_infids.items()):
        ax.semilogy(L_values, infids, 'o-', color=colors[ci],
                     linewidth=1.5, markersize=6, label=f'{it} iters')

    for L_bound, infid in direct_bounds.items():
        ax.axhline(y=infid, color='gray', linestyle='--', alpha=0.5,
                    label=f'Direct trunc (L={L_bound})')

    ax.set_xlabel('Number of Layers L', fontsize=12)
    ax.set_ylabel('Infidelity (1 − F)', fontsize=12)
    ax.set_title(f'Layer Scaling: N = {N}, Ising g = {g}', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'layer_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# =======================================================
# 主程序
# =======================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce Ben-Dov 2024 results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: smaller parameters for fast testing')
    parser.add_argument('--fig2a', action='store_true', help='Only Fig 2(a)')
    parser.add_argument('--xxz', action='store_true', help='Only XXZ')
    parser.add_argument('--layers', action='store_true', help='Only layer scaling')
    parser.add_argument('--all', action='store_true', help='Run all (default)')
    args = parser.parse_args()

    quick = args.quick
    run_all = args.all or (not args.fig2a and not args.xxz and not args.layers)

    print("=" * 60)
    print("Ben-Dov et al. (2024) — Reproduction Suite")
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 60)

    t_start = time.time()

    if run_all or args.fig2a:
        reproduce_fig2a(quick=quick)

    if run_all or args.layers:
        reproduce_layer_scaling(quick=quick)

    if run_all or args.xxz:
        reproduce_fig_xxz(quick=quick)

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")
