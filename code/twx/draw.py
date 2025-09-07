import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager
import os

# --- 1. 配置区域 ---
INITIAL_TEAMS_CSV = ""
OPTIMIZED_TEAMS_CSV = ""

UAV_NAMES = ['FY1', 'FY2', 'FY3']

# --- [终极解决方案] 直接指定字体文件路径 ---
# 这种方法可以绕过所有缓存和样式冲突问题
font_path = 'C:/Windows/Fonts/simhei.ttf' # SimHei (黑体) 在Windows下的标准路径

if os.path.exists(font_path):
    # 从字体文件创建一个 FontProperties 对象
    my_font = matplotlib.font_manager.FontProperties(fname=font_path)
    print(f"成功从路径 '{font_path}' 加载字体。")
else:
    print(f"错误：在路径 '{font_path}' 未找到字体文件。")
    print("请确认您的系统是否有该字体，或更换为其他可用字体的路径（如 'msyh.ttc' 对应微软雅黑）。")
    my_font = None # 设置为None，这样如果找不到字体程序会报错而不是继续

# ----------------------------------------------


def load_teams_from_csv(filename):
    """从指定的CSV文件中读取团队策略数据。"""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'。")
        return None, None

    teams_list = []
    scores = [] 

    for index, row in df.iterrows():
        team_data = {}
        for uav_name in UAV_NAMES:
            team_data[uav_name] = {
                'v': row[f'{uav_name}_v'],
                'theta_rad': np.radians(row[f'{uav_name}_theta_deg']),
                't_drop': row[f'{uav_name}_t_drop'],
                't_delay': row[f'{uav_name}_t_delay']
            }
        teams_list.append(team_data)
        
        if 'final_score' in df.columns:
            scores.append(row['final_score'])
        elif 'initial_total_score' in df.columns:
             scores.append(row['initial_total_score'])

    return teams_list, scores

def team_to_vector(team):
    """将团队策略字典转换为数值向量。"""
    vec = []
    for name in UAV_NAMES:
        strat = team[name]
        vec.extend([strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']])
    return vec


if __name__ == "__main__" and my_font is not None:
    print("--- 开始从CSV文件生成PCA可视化图 ---")

    initial_teams, _ = load_teams_from_csv(INITIAL_TEAMS_CSV)
    optimized_teams, optimized_scores = load_teams_from_csv(OPTIMIZED_TEAMS_CSV)

    if initial_teams is None or optimized_teams is None:
        print("因文件读取失败，程序已终止。")
    else:
        print(f"成功读取 {len(initial_teams)} 组初始团队数据。")
        print(f"成功读取 {len(optimized_teams)} 组优化后团队数据。")

        initial_vectors = np.array([team_to_vector(team) for team in initial_teams])
        final_vectors = np.array([team_to_vector(team) for team in optimized_teams])

        scaler = StandardScaler()
        all_vectors_scaled = scaler.fit_transform(np.vstack([initial_vectors, final_vectors]))

        pca = PCA(n_components=2)
        all_vectors_pca = pca.fit_transform(all_vectors_scaled)

        initial_pca = all_vectors_pca[:len(initial_teams)]
        final_pca = all_vectors_pca[len(initial_teams):]

        best_idx = np.argmax(optimized_scores)

        print("正在绘制PCA降维图...")
        # 为了避免样式覆盖字体，我们可以先不使用seaborn-v0_8-whitegrid，或者在设置样式后再设置字体
        # 但直接为每个元素指定字体属性是最保险的
        plt.style.use('default') # 使用默认样式，避免冲突
        plt.figure(figsize=(14, 10))
        plt.grid(True) # 手动开启网格

        for i in range(len(initial_teams)):
            plt.arrow(initial_pca[i, 0], initial_pca[i, 1],
                      final_pca[i, 0] - initial_pca[i, 0],
                      final_pca[i, 1] - initial_pca[i, 1],
                      head_width=0.08, fc='gray', ec='gray', alpha=0.5, length_includes_head=True)

        plt.scatter(initial_pca[:, 0], initial_pca[:, 1], c='dodgerblue', s=80, alpha=0.9, label='初始团队 ')
        plt.scatter(final_pca[:, 0], final_pca[:, 1], c='orangered', s=80, alpha=0.9, label='局部最优团队 ')
        plt.scatter(final_pca[best_idx, 0], final_pca[best_idx, 1],
                    c='gold', s=400, marker='*', edgecolors='black', linewidth=1.5, zorder=5,
                    label='最佳候选团队 ')

        # --- [核心修改] 为每个需要显示中文的元素，强制使用我们的字体 ---
        plt.title('12维协同策略空间的PCA降维可视化', fontproperties=my_font, fontsize=18, pad=20)
        plt.xlabel('主成分 1 ', fontproperties=my_font, fontsize=14)
        plt.ylabel('主成分 2 ', fontproperties=my_font, fontsize=14)
        plt.legend(prop=my_font, fontsize=12, loc='best') # 注意legend使用的是prop关键字
        
        output_filename = "optimization_journey_visualization.png"
        plt.savefig(output_filename, dpi=300)
        print(f"--- 可视化图已成功保存为 '{output_filename}' ---")
        
        plt.show()