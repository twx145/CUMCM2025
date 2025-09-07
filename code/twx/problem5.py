import numpy as np
import time
import itertools
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import multiprocessing
from tqdm import tqdm
from numba import njit

GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0
DROP_INTERVAL = 1.0  

UAV_NAMES = ['FY1', 'FY2']
UAV_INITIAL_POS = {
    'FY1': np.array([17800, 0, 1800], dtype=float),
    'FY2': np.array([12000, 1400, 1400], dtype=float),
}

NUM_INDIVIDUAL_STRATEGIES = 300 
NUM_TEAMS_TO_FORM = 300        
NUM_OPTIMIZATION_ROUNDS = 15     
OCCLUSION_THRESHOLD_PERCENT = 70.0 

missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * 300.0
missile_total_time = np.linalg.norm(false_target_pos - missile_initial_pos) / 300.0

@njit(fastmath=True)
def is_line_segment_intersecting_sphere_numba(p1, p2, sphere_center, sphere_radius):
    line_vec = p2 - p1
    point_vec = sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.sqrt(np.sum(point_vec**2)) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0.0, min(1.0, t))
    closest_point_on_line = p1 + t * line_vec
    dist_sq = np.sum((sphere_center - closest_point_on_line)**2)
    return dist_sq <= sphere_radius**2

def get_occlusion_timeline(team_strategies, uav_pos_dict, target_point, time_step=0.1):
    smoke_events, min_explode_time, max_end_time = [], float('inf'), float('-inf')

    for uav_name, strat in team_strategies.items():
        if uav_name not in uav_pos_dict: continue
        uav_pos = uav_pos_dict[uav_name]
        v, theta_rad = strat['v'], strat['theta_rad']
        uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
        
        drops = [strat.get('drop1'), strat.get('drop2'), strat.get('drop3')]
        if 't_drop' in strat: 
             drops = [(strat['t_drop'], strat['t_delay'])]

        for drop in drops:
            if not drop: continue
            t_drop, t_delay = drop
            t_explode = t_drop + t_delay
            p_drop = uav_pos + uav_velocity * t_drop
            p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
            smoke_events.append({'t_explode': t_explode, 'p_explode': p_explode})
            min_explode_time, max_end_time = min(min_explode_time, t_explode), max(max_end_time, t_explode + SMOKE_DURATION)

    if not smoke_events or min_explode_time > max_end_time: return 0

    timeline_len = int((max_end_time - min_explode_time) / time_step) + 1
    occlusion_timeline = np.zeros(timeline_len, dtype=bool)
    time_points = np.arange(min_explode_time, max_end_time, time_step)

    for t_idx, t_abs in enumerate(time_points):
        missile_pos = missile_initial_pos + missile_velocity_vector * t_abs
        for se in smoke_events:
            if se['t_explode'] <= t_abs < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -3.0 * (t_abs - se['t_explode'])])
                if is_line_segment_intersecting_sphere_numba(missile_pos, target_point, smoke_pos, SMOKE_RADIUS):
                    occlusion_timeline[t_idx] = True
                    break
    return np.sum(occlusion_timeline) * time_step

def stage1_worker(args):
    uav_name, uav_pos, n_top = args
    feasible_solutions = []
    search_explode_times = np.linspace(missile_total_time * 0.1, missile_total_time * 0.9, 300)
    search_delays = np.linspace(0, 8.0, 300)
    search_los_ratios = np.linspace(0, 0.9, 300)
    
    for t_explode in search_explode_times:
        for t_delay in search_delays:
            t_drop = t_explode - t_delay
            if t_drop <= 0.1: continue
            missile_pos_at_explode = missile_initial_pos + missile_velocity_vector * t_explode
            for los_ratio in search_los_ratios:
                los_vector = simple_target_point - missile_pos_at_explode
                ideal_explode_pos = missile_pos_at_explode + los_ratio * los_vector
                required_vel_3d = (ideal_explode_pos - uav_pos - np.array([0,0,-0.5*GRAVITY*t_delay**2])) / t_explode
                required_vel_xy = required_vel_3d[:2]
                required_speed = np.linalg.norm(required_vel_xy)
                if UAV_V_MIN <= required_speed <= UAV_V_MAX:
                    strat = { 'uav_name': uav_name, 'v': required_speed, 'theta_rad': np.arctan2(required_vel_xy[1], required_vel_xy[0]), 't_drop': t_drop, 't_delay': t_delay }
                    strat['occlusion_time'] = get_occlusion_timeline({uav_name: strat}, {uav_name: uav_pos}, simple_target_point)
                    if strat['occlusion_time'] > 1.0: feasible_solutions.append(strat)
    return uav_name, sorted(feasible_solutions, key=lambda x: x['occlusion_time'], reverse=True)[:n_top]

def stage1_individual_coarse_search_parallel(n_top):
    print("\n--- [阶段一] 开始：为所有无人机并行进行“单弹”策略全局粗搜 ---")
    tasks = [(name, UAV_INITIAL_POS[name], n_top) for name in UAV_NAMES]
    individual_strategies = {}
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(stage1_worker, tasks), total=len(tasks), desc="全局单弹策略粗搜"))
    for uav_name, strategies in results:
        individual_strategies[uav_name] = strategies
    print("--- [阶段一] 完成：所有无人机的顶尖单弹策略已生成 ---")
    return individual_strategies

def stage2_team_formation_and_expansion(individual_strats, n_teams):
    print(f"\n--- [阶段二] 开始：组合与扩展 {n_teams} 个“双机三弹”初始团队 ---")
    
    best_pairs = []
    strat_fy1 = individual_strats['FY1']
    strat_fy2 = individual_strats['FY2']
    
    all_combinations = []
    for s1 in tqdm(strat_fy1, desc="计算组合得分", leave=False):
        for s2 in strat_fy2:
            team = {'FY1': s1, 'FY2': s2}
            score = get_occlusion_timeline(team, UAV_INITIAL_POS, simple_target_point)
            all_combinations.append({'team': team, 'score': score})

    sorted_combinations = sorted(all_combinations, key=lambda x: x['score'], reverse=True)
    top_single_grenade_teams = [c['team'] for c in sorted_combinations[:n_teams]]

    expanded_teams = []
    for team in top_single_grenade_teams:
        expanded_team = {}
        for uav_name, strat in team.items():
            t_drop_center = strat['t_drop']
            base_delay = strat['t_delay']
            
            expanded_strat = {
                'v': strat['v'],
                'theta_rad': strat['theta_rad'],
                'drop1': (t_drop_center - DROP_INTERVAL, base_delay),
                'drop2': (t_drop_center, base_delay),
                'drop3': (t_drop_center + DROP_INTERVAL, base_delay),
            }
            if expanded_strat['drop1'][0] < 0.1:
                expanded_strat['drop1'] = (0.1, base_delay)
                expanded_strat['drop2'] = (0.1 + DROP_INTERVAL, base_delay)
                expanded_strat['drop3'] = (0.1 + 2 * DROP_INTERVAL, base_delay)
            
            expanded_team[uav_name] = expanded_strat
        
        expanded_team['occlusion_time'] = get_occlusion_timeline(expanded_team, UAV_INITIAL_POS, simple_target_point)
        expanded_teams.append(expanded_team)

    print(f"--- [阶段二] 完成：成功生成了 {len(expanded_teams)} 个初始“双机三弹”团队。---")
    return expanded_teams

def stage3_local_optimization_team(team_composition):
    params = {name: strat.copy() for name, strat in team_composition.items() if name in UAV_NAMES}
    current_max_time = team_composition.get('occlusion_time', get_occlusion_timeline(params, UAV_INITIAL_POS, simple_target_point))
    convergence_history = [current_max_time]

    for i in range(NUM_OPTIMIZATION_ROUNDS):
        last_occlusion_time = current_max_time
        for uav_name in UAV_NAMES:
            for key in ['v', 'theta_rad']:
                original_val = params[uav_name][key]
                step = 2.0 if key == 'v' else np.radians(4)
                search_range = np.linspace(original_val - step, original_val + step, 5)
                for test_val in search_range:
                    if key == 'v' and not (UAV_V_MIN <= test_val <= UAV_V_MAX): continue
                    test_params = {n: (s.copy() if n != uav_name else {**s, key: test_val}) for n, s in params.items()}
                    t = get_occlusion_timeline(test_params, UAV_INITIAL_POS, simple_target_point)
                    if t > current_max_time:
                        current_max_time = t
                        params[uav_name][key] = test_val
            
            for j in range(1, 4):
                drop_key = f'drop{j}'
                t_drop_base, t_delay_base = params[uav_name][drop_key]

                for t_drop_test in np.linspace(max(0.1, t_drop_base - 0.5), t_drop_base + 0.5, 5):
                    test_params = params.copy()
                    test_params[uav_name] = test_params[uav_name].copy()
                    test_params[uav_name][drop_key] = (t_drop_test, t_delay_base)
                    t = get_occlusion_timeline(test_params, UAV_INITIAL_POS, simple_target_point)
                    if t > current_max_time:
                        current_max_time = t
                        params[uav_name][drop_key] = (t_drop_test, t_delay_base)
                
                t_drop_base, t_delay_base = params[uav_name][drop_key] 
                for t_delay_test in np.linspace(max(0.1, t_delay_base - 0.5), t_delay_base + 0.5, 5):
                    test_params = params.copy()
                    test_params[uav_name] = test_params[uav_name].copy()
                    test_params[uav_name][drop_key] = (t_drop_base, t_delay_test)
                    t = get_occlusion_timeline(test_params, UAV_INITIAL_POS, simple_target_point)
                    if t > current_max_time:
                        current_max_time = t
                        params[uav_name][drop_key] = (t_drop_base, t_delay_test)

        convergence_history.append(current_max_time)
        if current_max_time - last_occlusion_time < 0.01:
            break
            
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
        strat = final_team[uav_name]; v, theta_rad = strat['v'], strat['theta_rad']
        uav_velocity = np.array([np.cos(theta_rad)*v, np.sin(theta_rad)*v, 0])
        final_team_details[uav_name] = {'v': v, 'theta_rad': theta_rad, 'drops': []}
        for i in range(1, 4):
            t_drop, t_delay = strat[f'drop{i}']
            t_explode = t_drop + t_delay
            p_drop = UAV_INITIAL_POS[uav_name] + uav_velocity * t_drop
            p_explode = p_drop + uav_velocity*t_delay + np.array([0,0,-0.5*GRAVITY*t_delay**2])
            smoke_events.append({'t_explode': t_explode, 'p_explode': p_explode})
            final_team_details[uav_name]['drops'].append({'t_drop': t_drop, 't_delay': t_delay, 't_explode': t_explode, 'p_drop': p_drop, 'p_explode': p_explode})
            min_explode_time, max_end_time = min(min_explode_time, t_explode), max(max_end_time, t_explode + SMOKE_DURATION)

    total_occluded_time = 0.0; time_step = 0.001
    for t in np.arange(min_explode_time, max_end_time, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t; is_occluded_this_step = False
        for se in smoke_events:
            if se['t_explode'] <= t < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -3.0 * (t - se['t_explode'])])
                occluded_lines_count = sum(1 for tp in target_points if is_line_segment_intersecting_sphere_numba(missile_pos, tp, smoke_pos, SMOKE_RADIUS))
                if occluded_lines_count >= required_occluded_count: is_occluded_this_step = True; break
        if is_occluded_this_step: total_occluded_time += time_step
    return total_occluded_time, final_team_details


def save_optimization_process_to_csv(optimized_teams, initial_teams, histories, filename="stage3_optimization_process_2UAV_6G.csv"):
    if not optimized_teams: return
    base_header = ['team_id', 'initial_score', 'final_score']
    for name in UAV_NAMES:
        base_header.extend([f'{name}_v', f'{name}_theta_deg', 
                            f'{name}_t_drop1', f'{name}_t_delay1',
                            f'{name}_t_drop2', f'{name}_t_delay2',
                            f'{name}_t_drop3', f'{name}_t_delay3'])
    iter_header = [f'iter_{i}_score' for i in range(NUM_OPTIMIZATION_ROUNDS + 1)]
    header = base_header + iter_header

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(header)
        for i, (opt_team, init_team, history) in enumerate(zip(optimized_teams, initial_teams, histories)):
            padded_history = history + [history[-1]] * (NUM_OPTIMIZATION_ROUNDS + 1 - len(history))
            row = [i + 1, f"{init_team['occlusion_time']:.4f}", f"{opt_team['occlusion_time']:.4f}"]
            for name in UAV_NAMES:
                strat = opt_team[name]
                row.extend([f"{strat['v']:.4f}", f"{np.degrees(strat['theta_rad']):.4f}",
                            f"{strat['drop1'][0]:.4f}", f"{strat['drop1'][1]:.4f}",
                            f"{strat['drop2'][0]:.4f}", f"{strat['drop2'][1]:.4f}",
                            f"{strat['drop3'][0]:.4f}", f"{strat['drop3'][1]:.4f}"])
            row.extend([f"{h:.4f}" for h in padded_history])
            writer.writerow(row)
    print(f"\n[日志] 阶段三：{len(optimized_teams)} 组团队的详细优化过程已保存到 {filename}")

def visualize_optimization_journey(initial_teams, optimized_teams):
    print("\n--- [可视化] 正在生成16维解空间的PCA降维图 ---")
    def team_to_vector(team):
        vec = []
        for name in UAV_NAMES:
            strat = team[name]
            vec.extend([strat['v'], strat['theta_rad'],
                        strat['drop1'][0], strat['drop1'][1],
                        strat['drop2'][0], strat['drop2'][1],
                        strat['drop3'][0], strat['drop3'][1]])
        return vec

    initial_vectors = np.array([team_to_vector(team) for team in initial_teams])
    final_vectors = np.array([team_to_vector(team) for team in optimized_teams])
    scaler = StandardScaler(); all_vectors_scaled = scaler.fit_transform(np.vstack([initial_vectors, final_vectors]))
    pca = PCA(n_components=2); all_vectors_pca = pca.fit_transform(all_vectors_scaled)
    initial_pca, final_pca = all_vectors_pca[:len(initial_teams)], all_vectors_pca[len(initial_teams):]
    best_idx = np.argmax([team['occlusion_time'] for team in optimized_teams])

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 10))
    for i in range(len(initial_teams)):
        plt.arrow(initial_pca[i, 0], initial_pca[i, 1], final_pca[i, 0] - initial_pca[i, 0], final_pca[i, 1] - initial_pca[i, 1], head_width=0.08, fc='gray', ec='gray', alpha=0.5, length_includes_head=True)
    plt.scatter(initial_pca[:, 0], initial_pca[:, 1], c='dodgerblue', s=80, alpha=0.9, label='初始团队 (Initial Teams)')
    plt.scatter(final_pca[:, 0], final_pca[:, 1], c='orangered', s=80, alpha=0.9, label='局部最优团队 (Local Optima)')
    plt.scatter(final_pca[best_idx, 0], final_pca[best_idx, 1], c='gold', s=400, marker='*', edgecolors='black', linewidth=1.5, zorder=5, label='最佳候选团队 (Best Candidate)')
    plt.title('16维协同策略空间的PCA降维可视化 (2 UAVs, 6 Grenades)', fontsize=18, pad=20)
    plt.xlabel('主成分 1 (Principal Component 1)', fontsize=14); plt.ylabel('主成分 2 (Principal Component 2)', fontsize=14)
    plt.legend(fontsize=12, loc='best'); plt.show()
if __name__ == "__main__":
    start_total_time = time.time()
    
    print("="*60)
    print("  双机协同六弹干扰问题求解器启动 (CPU并行+Numba+P3/P4逻辑融合)")
    print("="*60)
    
    individual_strategies = stage1_individual_coarse_search_parallel(n_top=NUM_INDIVIDUAL_STRATEGIES)
    
    initial_teams = stage2_team_formation_and_expansion(individual_strategies, NUM_TEAMS_TO_FORM)

    if not initial_teams:
        print("错误：阶段二未能组合出任何团队。")
    else:
        print(f"\n--- [阶段三] 开始：对 {len(initial_teams)} 个团队进行局部协同调优 ---")
        optimized_teams, histories = [], []
        for i, team in enumerate(tqdm(initial_teams, desc="局部协同调优")):
            optimized_team, history = stage3_local_optimization_team(team)
            optimized_teams.append(optimized_team)
            histories.append(history)
        
        save_optimization_process_to_csv(optimized_teams, initial_teams, histories)
        visualize_optimization_journey(initial_teams, optimized_teams)
        
        champion_team = max(optimized_teams, key=lambda x: x['occlusion_time'])
        
        print("\n--- [阶段四] 开始：对冠军团队进行最终高精度验证 ---")
        final_time, final_details = stage4_final_validation_team(champion_team, threshold_percent=OCCLUSION_THRESHOLD_PERCENT)
        
        print("\n" + "="*70)
        print("            最优“双机六弹”协同干扰策略及最终结果")
        print("="*70)
        print(f"\n搜索概况: 从 {len(initial_teams)} 个高质量团队组合出发，找到的最佳协同策略如下。")
        for uav_name in UAV_NAMES:
            details = final_details[uav_name]
            print(f"\n--- 无人机 {uav_name} 统一飞行策略 ---")
            print(f"  飞行速度 (v)      : {details['v']:.4f} m/s")
            print(f"  飞行方向 (θ)      : {np.degrees(details['theta_rad']):.4f} 度")
            for i, drop_info in enumerate(details['drops']):
                print(f"\n  --- {uav_name} 第 {i+1} 枚烟幕弹 ---")
                print(f"    投放时间 (t_drop)   : {drop_info['t_drop']:.4f} 秒")
                print(f"    引信延迟 (t_delay)    : {drop_info['t_delay']:.4f} 秒")
                print(f"    投放点坐标 (P_drop)   : ({drop_info['p_drop'][0]:.2f}, {drop_info['p_drop'][1]:.2f}, {drop_info['p_drop'][2]:.2f})")
                print(f"    起爆点坐标 (P_explode): ({drop_info['p_explode'][0]:.2f}, {drop_info['p_explode'][1]:.2f}, {drop_info['p_explode'][2]:.2f})")

        print("\n" + "-"*70)
        print("--- 最终性能评估 ---")
        print(f"  遮蔽有效性阈值          : {OCCLUSION_THRESHOLD_PERCENT:.1f}%")
        print(f"  最终高精度总有效遮蔽时长: {final_time:.4f} 秒")
        print("="*70)

    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")