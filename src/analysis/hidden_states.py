"""
Hidden States分析模块
"""

import numpy as np
from typing import List, Dict

from ..util import cosine_similarity, kl_divergence_from_hidden


def analyze_hidden_states(hidden_states: List[Dict]) -> List[Dict]:
    """
    分析hidden states序列，计算相邻token之间的差异

    对每个token计算：
    - delta = h_t - h_{t-1}
    - delta的模长
    - delta与h_{t-1}的余弦相似度（表示变化方向与原方向的关系）
    - h_t与h_{t-1}的余弦相似度
    - h_t与h_{t-1}的KL散度
    - h_t的模长
    - delta在h_{t-1}方向上的投影长度（切向分量）
    - delta垂直于h_{t-1}的分量的模长（法向分量）

    Args:
        hidden_states: hidden state数据列表，每个元素包含:
            - hidden_state: 向量
            - token_id: token ID
            - token_text: token文本

    Returns:
        分析结果列表
    """
    results = []

    for i in range(len(hidden_states)):
        hs = hidden_states[i]
        h_t = np.array(hs["hidden_state"])

        info = {
            "index": i,
            "token_id": hs["token_id"],
            "token_text": hs["token_text"],
            "h_t_norm": float(np.linalg.norm(h_t)),
        }

        if i > 0:
            h_prev = np.array(hidden_states[i - 1]["hidden_state"])
            delta = h_t - h_prev

            delta_norm = float(np.linalg.norm(delta))
            h_prev_norm = float(np.linalg.norm(h_prev))

            # delta与h_{t-1}的余弦相似度
            cos_delta_hprev = cosine_similarity(delta, h_prev)

            # h_t与h_{t-1}的余弦相似度
            cos_ht_hprev = cosine_similarity(h_t, h_prev)

            # KL散度
            kl_div = kl_divergence_from_hidden(h_t, h_prev)

            # 计算delta在h_{t-1}方向上的投影（切向分量）
            if h_prev_norm > 0:
                # 投影长度 = delta · (h_prev / |h_prev|)
                tangent_proj = float(np.dot(delta, h_prev) / h_prev_norm)
                # 法向分量的模长
                tangent_vec = tangent_proj * (h_prev / h_prev_norm)
                normal_vec = delta - tangent_vec
                normal_norm = float(np.linalg.norm(normal_vec))
            else:
                tangent_proj = 0.0
                normal_norm = delta_norm

            # 相对变化率
            relative_change = delta_norm / h_prev_norm if h_prev_norm > 0 else 0.0

            info.update(
                {
                    "delta_norm": delta_norm,
                    "cos_delta_hprev": cos_delta_hprev,
                    "cos_ht_hprev": cos_ht_hprev,
                    "kl_divergence": kl_div,
                    "tangent_proj": tangent_proj,  # 切向投影长度（正=同向增长，负=反向）
                    "normal_norm": normal_norm,  # 法向分量模长（方向变化大小）
                    "relative_change": relative_change,  # 相对变化率
                }
            )
        else:
            info.update(
                {
                    "delta_norm": None,
                    "cos_delta_hprev": None,
                    "cos_ht_hprev": None,
                    "kl_divergence": None,
                    "tangent_proj": None,
                    "normal_norm": None,
                    "relative_change": None,
                }
            )

        results.append(info)

    return results


def svd_analysis(hidden_states: List[Dict], top_k: int = 10, center: bool = False) -> Dict:
    """
    对hidden states进行SVD奇异值分解

    Args:
        hidden_states: hidden state数据列表，每个元素包含:
            - hidden_state: 向量
            - token_id: token ID
            - token_text: token文本
        top_k: 返回前k个奇异值
        center: 是否对数据减去均值（中心化）

    Returns:
        包含SVD分析结果的字典:
            - singular_values: 前top_k个奇异值
            - total_variance: 总方差（奇异值平方和）
            - explained_variance_ratio: 前top_k个奇异值解释的方差比例
            - centered: 是否进行了中心化
            - matrix_shape: 数据矩阵的形状 (n_samples, n_features)
    """
    # 构建数据矩阵，每行是一个hidden state向量
    matrix = np.array([hs["hidden_state"] for hs in hidden_states])
    
    # 中心化处理
    if center:
        mean_vec = np.mean(matrix, axis=0)
        matrix = matrix - mean_vec
    
    # SVD分解
    # matrix = U @ diag(S) @ Vh
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # 计算总方差和解释方差比例
    total_variance = np.sum(S ** 2)
    explained_variance = S[:top_k] ** 2
    explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else np.zeros(top_k)
    cumulative_ratio = np.cumsum(explained_variance_ratio)
    
    return {
        "singular_values": S[:top_k].tolist(),
        "all_singular_values": S.tolist(),
        "total_variance": float(total_variance),
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "cumulative_variance_ratio": cumulative_ratio.tolist(),
        "centered": center,
        "matrix_shape": matrix.shape,
    }


def print_svd_results(svd_result: Dict, title: str = "SVD分析结果"):
    """
    打印SVD分析结果

    Args:
        svd_result: svd_analysis函数返回的结果字典
        title: 打印标题
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"数据矩阵形状: {svd_result['matrix_shape']}")
    print(f"是否中心化: {'是' if svd_result['centered'] else '否'}")
    print(f"总方差: {svd_result['total_variance']:.4f}")
    print(f"\n前{len(svd_result['singular_values'])}个奇异值:")
    print("-" * 60)
    print(f"{'序号':>4} {'奇异值':>15} {'方差占比':>12} {'累计方差占比':>12}")
    print("-" * 60)
    for i, (sv, ratio, cum_ratio) in enumerate(zip(
        svd_result['singular_values'],
        svd_result['explained_variance_ratio'],
        svd_result['cumulative_variance_ratio']
    )):
        print(f"{i+1:>4} {sv:>15.4f} {ratio*100:>11.2f}% {cum_ratio*100:>11.2f}%")
    print("-" * 60)


def visualize_svd_trajectory(
    hidden_states: List[Dict],
    output_path: str = None,
    title: str = "Hidden States SVD Projection Trajectory",
    figsize: tuple = (12, 10),
    show_arrows: bool = True,
    arrow_interval: int = 1,
    colormap: str = "viridis",
):
    """
    将hidden states投影到SVD前两个主成分上并可视化演进路径

    Args:
        hidden_states: hidden state数据列表
        output_path: 图片保存路径，如果为None则显示图片
        title: 图表标题
        figsize: 图表大小
        show_arrows: 是否显示箭头表示方向
        arrow_interval: 箭头间隔（每隔多少个点画一个箭头）
        colormap: 颜色映射名称

    Returns:
        投影后的坐标数组 (n_samples, 2)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.collections import LineCollection
    
    # 构建数据矩阵
    matrix = np.array([hs["hidden_state"] for hs in hidden_states])
    
    # SVD分解（不中心化，使用原始数据）
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # 投影到前两个主成分
    # 投影坐标 = matrix @ Vh.T[:, :2] = U @ diag(S) @ Vh @ Vh.T[:, :2]
    # 简化为 U[:, :2] @ diag(S[:2])
    projected = U[:, :2] * S[:2]
    
    # 计算方差解释比例
    total_variance = np.sum(S ** 2)
    var_ratio_1 = (S[0] ** 2) / total_variance * 100
    var_ratio_2 = (S[1] ** 2) / total_variance * 100
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 颜色映射：按时间步着色
    n_points = len(projected)
    colors = np.linspace(0, 1, n_points)
    cmap = cm.get_cmap(colormap)
    
    # 绘制轨迹线（使用LineCollection实现渐变色）
    points = projected.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, alpha=0.7, linewidth=1.5)
    lc.set_array(colors[:-1])
    ax.add_collection(lc)
    
    # 绘制散点
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1],
        c=colors, cmap=cmap, s=30, alpha=0.8, edgecolors='white', linewidth=0.5
    )
    
    # 标记起点和终点
    ax.scatter(projected[0, 0], projected[0, 1], c='green', s=200, marker='o', 
               label='Start', zorder=5, edgecolors='white', linewidth=2)
    ax.scatter(projected[-1, 0], projected[-1, 1], c='red', s=200, marker='s', 
               label='End', zorder=5, edgecolors='white', linewidth=2)
    
    # 绘制箭头表示方向
    if show_arrows and n_points > 1:
        for i in range(0, n_points - 1, arrow_interval):
            dx = projected[i + 1, 0] - projected[i, 0]
            dy = projected[i + 1, 1] - projected[i, 1]
            # 只在移动距离足够大时画箭头
            if np.sqrt(dx**2 + dy**2) > 0.01 * max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1])):
                ax.annotate('', xy=(projected[i + 1, 0], projected[i + 1, 1]),
                           xytext=(projected[i, 0], projected[i, 1]),
                           arrowprops=dict(arrowstyle='->', color=cmap(colors[i]), alpha=0.6, lw=1))
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, label='Time Step (normalized)')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0', f'{n_points//4}', f'{n_points//2}', f'{3*n_points//4}', f'{n_points-1}'])
    
    # 设置标签和标题
    ax.set_xlabel(f'PC1 (Variance Ratio: {var_ratio_1:.2f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 (Variance Ratio: {var_ratio_2:.2f}%)', fontsize=12)
    ax.set_title(f'{title}\n({n_points} tokens)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围，留出一定边距
    x_margin = 0.1 * np.ptp(projected[:, 0])
    y_margin = 0.1 * np.ptp(projected[:, 1])
    ax.set_xlim(projected[:, 0].min() - x_margin, projected[:, 0].max() + x_margin)
    ax.set_ylim(projected[:, 1].min() - y_margin, projected[:, 1].max() + y_margin)
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ SVD轨迹图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return projected

def visualize_svd_trajectory_3d(
    hidden_states: List[Dict],
    output_path: str = None,
    title: str = "Hidden States SVD 3D Projection Trajectory",
    figsize: tuple = (12, 10),
    colormap: str = "viridis",
    elev: float = 30,
    azim: float = 45,
):
    """
    将hidden states投影到SVD前三个主成分上并可视化3D演进路径

    Args:
        hidden_states: hidden state数据列表
        output_path: 图片保存路径，如果为None则显示图片
        title: 图表标题
        figsize: 图表大小
        colormap: 颜色映射名称
        elev: 3D视图仰角
        azim: 3D视图方位角

    Returns:
        投影后的坐标数组 (n_samples, 3)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    
    # 构建数据矩阵
    matrix = np.array([hs["hidden_state"] for hs in hidden_states])
    
    # SVD分解（不中心化，使用原始数据）
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # 投影到前三个主成分
    projected = U[:, :3] * S[:3]
    
    # 计算方差解释比例
    total_variance = np.sum(S ** 2)
    var_ratio_1 = (S[0] ** 2) / total_variance * 100
    var_ratio_2 = (S[1] ** 2) / total_variance * 100
    var_ratio_3 = (S[2] ** 2) / total_variance * 100
    
    # 创建3D图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色映射：按时间步着色
    n_points = len(projected)
    colors = np.linspace(0, 1, n_points)
    cmap = cm.get_cmap(colormap)
    
    # 绘制3D轨迹线（使用Line3DCollection实现渐变色）
    points = projected.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, alpha=0.7, linewidth=1.5)
    lc.set_array(colors[:-1])
    ax.add_collection3d(lc)
    
    # 绘制散点
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1], projected[:, 2],
        c=colors, cmap=cmap, s=30, alpha=0.8, edgecolors='white', linewidth=0.5
    )
    
    # 标记起点和终点
    ax.scatter(projected[0, 0], projected[0, 1], projected[0, 2], 
               c='green', s=200, marker='o', label='Start', zorder=5, edgecolors='white', linewidth=2)
    ax.scatter(projected[-1, 0], projected[-1, 1], projected[-1, 2], 
               c='red', s=200, marker='s', label='End', zorder=5, edgecolors='white', linewidth=2)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, label='Time Step (normalized)', shrink=0.6, pad=0.1)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0', f'{n_points//4}', f'{n_points//2}', f'{3*n_points//4}', f'{n_points-1}'])
    
    # 设置标签和标题
    ax.set_xlabel(f'PC1 ({var_ratio_1:.2f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_ratio_2:.2f}%)', fontsize=10)
    ax.set_zlabel(f'PC3 ({var_ratio_3:.2f}%)', fontsize=10)
    ax.set_title(f'{title}\n({n_points} tokens, Total Variance: {var_ratio_1+var_ratio_2+var_ratio_3:.2f}%)', fontsize=12)
    ax.legend(loc='upper right')
    
    # 设置视角
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ SVD 3D轨迹图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return projected