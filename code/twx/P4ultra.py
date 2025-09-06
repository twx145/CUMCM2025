import numpy as np
import time
import itertools
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 0. 基础设置与可调参数 ---
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0

# --- 第四问特定参数 ---
UAV_NAMES = ['FY1', 'FY2', 'FY3'] # 确保顺序一致性
UAV_INITIAL_POS = {
    'FY1': np.array([17800, 0, 1800], dtype=float),
    'FY2': np.array([12000, 1400, 1400], dtype=float),
    'FY3': np.array([6000, -3000, 700], dtype=float),
}
NUM_INDIVIDUAL_STRATEGIES = 50
NUM_TEAMS_TO_FORM = 50
NUM_OPTIMIZATION_ROUNDS = 10

# 最终验证的遮蔽阈值
OCCLUSION_THRESHOLD_PERCENT = 70.0

# 全局实体位置 (与之前相同)
missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * 300.0
missile_total_time = np.linalg.norm(false_target_pos - missile_initial_pos) / 300.0

# --- 辅助函数与核心计算函数 (与之前版本完全相同) ---
# ... (此处省略 is_line_segment_intersecting_sphere, get_occlusion_timeline 等函数)
def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius):
    if np.any(np.isnan(sphere_center)): return False
    line_vec, point_vec = p2 - p1, sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.linalg.norm(point_vec) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    closest_point = p1 + np.clip(t, 0, 1) * line_vec
    return np.linalg.norm(sphere_center - closest_point) <= sphere_radius

def get_occlusion_timeline(strategies, uav_pos_dict, target_point, time_step=0.1):
    if not strategies: return 0
    smoke_events, min_explode_time, max_end_time = [], float('inf'), float('-inf')
    for strat in strategies:
        uav_pos = uav_pos_dict[strat['uav_name']]
        v, theta_rad, t_drop, t_delay = strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']
        uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
        t_explode = t_drop + t_delay
        p_drop = uav_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        smoke_events.append({'t_explode': t_explode, 'p_explode': p_explode})
        min_explode_time, max_end_time = min(min_explode_time, t_explode), max(max_end_time, t_explode + SMOKE_DURATION)
    if min_explode_time > max_end_time: return 0
    timeline_len = int((max_end_time - min_explode_time) / time_step) + 1
    occlusion_timeline = np.zeros(timeline_len, dtype=bool)
    time_points = np.arange(min_explode_time, max_end_time, time_step)
    for t_idx, t_abs in enumerate(time_points):
        missile_pos = missile_initial_pos + missile_velocity_vector * t_abs
        for se in smoke_events:
            if se['t_explode'] <= t_abs < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -3.0 * (t_abs - se['t_explode'])])
                if is_line_segment_intersecting_sphere(missile_pos, target_point, smoke_pos, SMOKE_RADIUS):
                    occlusion_timeline[t_idx] = True; break
    return np.sum(occlusion_timeline) * time_step

# --- 四阶段求解函数 (与之前版本完全相同) ---
# ... (此处省略 stage1, stage2, stage3, stage4, 和日志记录函数)
def stage1_individual_coarse_search(uav_name, uav_pos, n_top):
    print(f"  - [阶段1] 正在为 {uav_name} 进行全局粗搜...")
    feasible_solutions = []
    search_explode_times = np.linspace(missile_total_time * 0, missile_total_time * 0.9, 150)
    search_delays = np.linspace(0, 8.0, 100)
    search_los_ratios = np.linspace(0, 0.9, 50)
    for t_explode in search_explode_times:
        for t_delay in search_delays:
            for los_ratio in search_los_ratios:
                if t_explode <= t_delay: continue
                missile_pos_at_explode = missile_initial_pos + missile_velocity_vector * t_explode
                los_vector = simple_target_point - missile_pos_at_explode
                ideal_explode_pos = missile_pos_at_explode + los_ratio * los_vector
                t_drop = t_explode - t_delay
                if t_drop <= 0: continue
                required_vel_3d = (ideal_explode_pos - uav_pos - np.array([0,0,-0.5*GRAVITY*t_delay**2])) / t_explode
                required_vel_xy = required_vel_3d[:2]
                required_speed = np.linalg.norm(required_vel_xy)
                if UAV_V_MIN <= required_speed <= UAV_V_MAX:
                    strat = { 'uav_name': uav_name, 'v': required_speed, 'theta_rad': np.arctan2(required_vel_xy[1], required_vel_xy[0]), 't_drop': t_drop, 't_delay': t_delay }
                    strat['occlusion_time'] = get_occlusion_timeline([strat], {uav_name: uav_pos}, simple_target_point)
                    if strat['occlusion_time'] > 0: feasible_solutions.append(strat)
    return sorted(feasible_solutions, key=lambda x: x['occlusion_time'], reverse=True)[:n_top]

def stage2_greedy_combination(individual_strats, n_teams):
    print(f"\n--- [阶段2] 开始：基于边际增益的贪心算法组合 {n_teams} 个团队 ---")
    all_strats_sorted = sorted(itertools.chain(*individual_strats.values()), key=lambda x: x['occlusion_time'], reverse=True)
    teams = []
    for i in range(min(n_teams, len(all_strats_sorted))):
        seed_strat = all_strats_sorted[i]
        current_uavs, team_strats = [seed_strat['uav_name']], [seed_strat]
        for _ in range(2):
            best_gain, best_strat_to_add = -1, None
            for uav_name_to_add in UAV_NAMES:
                if uav_name_to_add in current_uavs: continue
                for strat_to_add in individual_strats[uav_name_to_add]:
                    current_time = get_occlusion_timeline(team_strats + [strat_to_add], UAV_INITIAL_POS, simple_target_point)
                    if current_time > best_gain:
                        best_gain, best_strat_to_add = current_time, strat_to_add
            if best_strat_to_add: team_strats.append(best_strat_to_add); current_uavs.append(best_strat_to_add['uav_name'])
        if len(team_strats) == 3:
            team = {s['uav_name']: s for s in team_strats}
            team['occlusion_time'] = get_occlusion_timeline(team_strats, UAV_INITIAL_POS, simple_target_point)
            teams.append(team)
    unique_teams = []
    for team in sorted(teams, key=lambda x: x['occlusion_time'], reverse=True):
        if not any(all(abs(team[n]['t_drop'] - ut[n]['t_drop']) < 0.2 for n in UAV_NAMES) for ut in unique_teams):
            unique_teams.append(team)
    print(f"阶段2完成。成功组合了 {len(unique_teams)} 个独特的团队。")
    return unique_teams

def stage3_local_optimization_team(team_composition):
    params = {name: strat.copy() for name, strat in team_composition.items() if name in UAV_NAMES}
    current_max_time = team_composition.get('occlusion_time', get_occlusion_timeline(list(params.values()), UAV_INITIAL_POS, simple_target_point))
    convergence_history = [current_max_time]
    for i in range(NUM_OPTIMIZATION_ROUNDS):
        last_occlusion_time = current_max_time
        for uav_name in UAV_NAMES:
            for key in ['v', 'theta_rad', 't_drop', 't_delay']:
                original_val = params[uav_name][key]
                step = 2.5 if key == 'v' else np.radians(5) if key == 'theta_rad' else 0.5
                search_range = np.linspace(original_val - step, original_val + step, 5)
                for test_val in search_range:
                    test_params_list = [{**params[name], **({key: test_val} if name == uav_name else {})} for name in UAV_NAMES]
                    if key == 'v' and not (UAV_V_MIN <= test_val <= UAV_V_MAX): continue
                    t = get_occlusion_timeline(test_params_list, UAV_INITIAL_POS, simple_target_point)
                    if t > current_max_time: current_max_time, params[uav_name][key] = t, test_val
        convergence_history.append(current_max_time)
        if current_max_time - last_occlusion_time < 0.01: break
    params['occlusion_time'] = current_max_time
    return params, convergence_history
    
def stage4_final_validation_team(final_team, n_points=1000, threshold_percent=70.0):
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points)); target_points = []
    for _ in range(int(n_points * 0.6)):
        theta, z = 2*np.pi*np.random.rand(), true_target_height*np.random.rand(); target_points.append([true_target_base_center[0]+7*np.cos(theta), true_target_base_center[1]+7*np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = 7*np.sqrt(np.random.rand()), 2*np.pi*np.random.rand(); z = 0 if np.random.rand() < 0.5 else true_target_height; target_points.append([true_target_base_center[0]+r*np.cos(theta), true_target_base_center[1]+r*np.sin(theta), z])
    target_points = np.array(target_points)
    smoke_events, final_team_details = [], {}
    min_explode_time, max_end_time = float('inf'), float('-inf')
    for uav_name in UAV_NAMES:
        strat = final_team[uav_name]; v, theta_rad, t_drop, t_delay = strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']
        uav_velocity = np.array([np.cos(theta_rad)*v, np.sin(theta_rad)*v, 0]); t_explode = t_drop + t_delay
        p_drop = UAV_INITIAL_POS[uav_name] + uav_velocity * t_drop; p_explode = p_drop + uav_velocity*t_delay + np.array([0,0,-0.5*GRAVITY*t_delay**2])
        smoke_events.append({'t_explode': t_explode, 'p_explode': p_explode}); final_team_details[uav_name] = {**strat, 'p_drop': p_drop, 'p_explode': p_explode, 't_explode': t_explode}
        min_explode_time, max_end_time = min(min_explode_time, t_explode), max(max_end_time, t_explode + SMOKE_DURATION)
    total_occluded_time = 0.0; time_step = 0.001
    for t in np.arange(min_explode_time, max_end_time, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t; is_occluded_this_step = False
        for se in smoke_events:
            if se['t_explode'] <= t < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -3.0 * (t - se['t_explode'])])
                occluded_lines_count = sum(1 for tp in target_points if is_line_segment_intersecting_sphere(missile_pos, tp, smoke_pos, SMOKE_RADIUS))
                if occluded_lines_count >= required_occluded_count: is_occluded_this_step = True; break
        if is_occluded_this_step: total_occluded_time += time_step
    return total_occluded_time, final_team_details

def save_greedy_teams_to_csv(teams, filename="stage2_greedy_selections.csv"):
    if not teams: return
    header = ['team_id', 'initial_total_score']
    for name in UAV_NAMES: header.extend([f'{name}_v', f'{name}_theta_deg', f'{name}_t_drop', f'{name}_t_delay'])
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, team in enumerate(teams):
            row = [i + 1, f"{team['occlusion_time']:.4f}"]
            for name in UAV_NAMES: row.extend([f"{team[name]['v']:.4f}", f"{np.degrees(team[name]['theta_rad']):.4f}", f"{team[name]['t_drop']:.4f}", f"{team[name]['t_delay']:.4f}"])
            writer.writerow(row)
    print(f"\n[日志] 阶段2：{len(teams)} 组贪心算法选择的团队参数已保存到 {filename}")

def save_optimization_process_to_csv(optimized_teams, initial_teams, histories, filename="stage3_optimization_process.csv"):
    if not optimized_teams: return
    base_header = ['team_id', 'initial_score', 'final_score']
    for name in UAV_NAMES: base_header.extend([f'{name}_v', f'{name}_theta_deg', f'{name}_t_drop', f'{name}_t_delay'])
    iter_header = [f'iter_{i}_score' for i in range(NUM_OPTIMIZATION_ROUNDS + 1)]
    header = base_header + iter_header
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(header)
        for i, (opt_team, init_team, history) in enumerate(zip(optimized_teams, initial_teams, histories)):
            padded_history = history + [history[-1]] * (NUM_OPTIMIZATION_ROUNDS + 1 - len(history))
            row = [i + 1, f"{init_team['occlusion_time']:.4f}", f"{opt_team['occlusion_time']:.4f}"]
            for name in UAV_NAMES: strat = opt_team[name]; row.extend([f"{strat['v']:.4f}", f"{np.degrees(strat['theta_rad']):.4f}", f"{strat['t_drop']:.4f}", f"{strat['t_delay']:.4f}"])
            row.extend([f"{h:.4f}" for h in padded_history])
            writer.writerow(row)
    print(f"[日志] 阶段3：{len(optimized_teams)} 组团队的详细优化过程已保存到 {filename}")

# --- 新增：PCA 可视化函数 ---
def visualize_optimization_journey(initial_teams, optimized_teams):
    """
    使用PCA将12维的团队策略向量降维到2D并可视化优化过程。
    """
    print("\n--- [可视化] 正在生成12维解空间的PCA降维图 ---")
    
    def team_to_vector(team):
        # 确保始终按 UAV_NAMES 的顺序提取参数，保证向量的一致性
        vec = []
        for name in UAV_NAMES:
            strat = team[name]
            vec.extend([strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']])
        return vec

    initial_vectors = np.array([team_to_vector(team) for team in initial_teams])
    final_vectors = np.array([team_to_vector(team) for team in optimized_teams])
    
    # 对所有数据进行标准化
    scaler = StandardScaler()
    all_vectors_scaled = scaler.fit_transform(np.vstack([initial_vectors, final_vectors]))
    
    # 应用PCA
    pca = PCA(n_components=2)
    all_vectors_pca = pca.fit_transform(all_vectors_scaled)
    
    # 分离出初始和优化后的点
    initial_pca = all_vectors_pca[:len(initial_teams)]
    final_pca = all_vectors_pca[len(initial_teams):]
    
    # 找到最佳解的索引
    best_idx = np.argmax([team['occlusion_time'] for team in optimized_teams])
    
    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 10))
    
    # 绘制优化的路径箭头
    for i in range(len(initial_teams)):
        plt.arrow(initial_pca[i, 0], initial_pca[i, 1], 
                  final_pca[i, 0] - initial_pca[i, 0], 
                  final_pca[i, 1] - initial_pca[i, 1], 
                  head_width=0.08, fc='gray', ec='gray', alpha=0.5, length_includes_head=True)

    # 绘制散点
    plt.scatter(initial_pca[:, 0], initial_pca[:, 1], c='dodgerblue', s=80, alpha=0.9, label='初始团队 (Greedy Selections)')
    plt.scatter(final_pca[:, 0], final_pca[:, 1], c='orangered', s=80, alpha=0.9, label='局部最优团队 (Local Optima)')
    
    # 高亮最佳解
    plt.scatter(final_pca[best_idx, 0], final_pca[best_idx, 1], 
                c='gold', s=400, marker='*', edgecolors='black', linewidth=1.5, zorder=5,
                label='最佳候选团队 (Best Candidate)')

    plt.title('12维协同策略空间的PCA降维可视化', fontsize=18, pad=20)
    plt.xlabel('主成分 1 (Principal Component 1)', fontsize=14)
    plt.ylabel('主成分 2 (Principal Component 2)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.show()

# --- 主程序 (集成PCA可视化调用) ---
if __name__ == "__main__":
    start_total_time = time.time()
    
    print("="*60)
    print("        第四问：三机协同干扰问题求解器启动 (带详细日志和PCA可视化)")
    print("="*60)
    
    individual_strategies = {name: stage1_individual_coarse_search(name, UAV_INITIAL_POS[name], NUM_INDIVIDUAL_STRATEGIES) for name in UAV_NAMES}
    initial_teams = stage2_greedy_combination(individual_strategies, NUM_TEAMS_TO_FORM)
    
    save_greedy_teams_to_csv(initial_teams)

    if not initial_teams:
        print("错误：阶段二未能组合出任何团队。")
    else:
        print(f"\n--- [阶段3] 开始：对 {len(initial_teams)} 个团队进行局部协同调优 ---")
        optimized_teams, histories = [], []
        for i, team in enumerate(initial_teams):
            print(f"  - 正在优化团队 {i+1}/{len(initial_teams)} (初始时长: {team['occlusion_time']:.2f}s)...", end='')
            optimized_team, history = stage3_local_optimization_team(team)
            optimized_teams.append(optimized_team)
            histories.append(history)
            print(f" 优化后时长: {optimized_team['occlusion_time']:.4f}s")
        
        save_optimization_process_to_csv(optimized_teams, initial_teams, histories)

        # --- 新增：调用PCA可视化 ---
        visualize_optimization_journey(initial_teams, optimized_teams)
        
        champion_team = max(optimized_teams, key=lambda x: x['occlusion_time'])
        
        print("\n--- [阶段4] 开始：对冠军团队进行最终高精度验证 ---")
        final_time, final_details = stage4_final_validation_team(champion_team, threshold_percent=OCCLUSION_THRESHOLD_PERCENT)
        
        # 最终结果打印 (不变)
        print("\n" + "="*60)
        # ... (省略最终结果打印，与上一版完全相同)
        print("        第四问：最优协同干扰策略及最终结果")
        print("="*60)
        print(f"\n搜索概况: 从 {len(initial_teams)} 个高质量团队组合出发，找到的最佳策略如下。")
        for uav_name in UAV_NAMES:
            strat = final_details[uav_name]
            print(f"\n--- 无人机 {uav_name} 策略 ---")
            print(f"  飞行速度 (v)      : {strat['v']:.4f} m/s")
            print(f"  飞行方向 (θ)      : {np.degrees(strat['theta_rad']):.4f} 度")
            print(f"  投放时间 (t_drop)   : {strat['t_drop']:.4f} 秒")
            print(f"  引信延迟 (t_delay)    : {strat['t_delay']:.4f} 秒")
            print(f"  投放点坐标 (P_drop)   : ({strat['p_drop'][0]:.2f}, {strat['p_drop'][1]:.2f}, {strat['p_drop'][2]:.2f})")
            print(f"  起爆点坐标 (P_explode): ({strat['p_explode'][0]:.2f}, {strat['p_explode'][1]:.2f}, {strat['p_explode'][2]:.2f})")
        print("\n--- 最终性能评估 ---")
        print(f"  遮蔽有效性阈值          : {OCCLUSION_THRESHOLD_PERCENT:.1f}%")
        print(f"  最终高精度总有效遮蔽时长: {final_time:.4f} 秒")
        print("="*60)

    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")