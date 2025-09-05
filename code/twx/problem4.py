import numpy as np
import time
import itertools

# --- 0. 基础设置与可调参数 ---
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0

# --- 第四问特定参数 ---
UAV_NAMES = ['FY1', 'FY2', 'FY3']
UAV_INITIAL_POS = {
    'FY1': np.array([17800, 0, 1800], dtype=float),
    'FY2': np.array([12000, 1400, 1400], dtype=float),
    'FY3': np.array([6000, -3000, 700], dtype=float),
}
# 阶段一：为每架无人机保留的个体最优策略数
NUM_INDIVIDUAL_STRATEGIES = 5
# 阶段二：使用贪心算法构建的团队组合数
NUM_TEAMS_TO_FORM = 5
# 阶段三：局部优化的迭代轮数
NUM_OPTIMIZATION_ROUNDS = 2

# 最终验证的遮蔽阈值
OCCLUSION_THRESHOLD_PERCENT = 70.0

# 全局实体位置
missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * 300.0
missile_total_time = np.linalg.norm(false_target_pos - missile_initial_pos) / 300.0


# --- 辅助函数 ---
def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius):
    if np.any(np.isnan(sphere_center)): return False
    line_vec, point_vec = p2 - p1, sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.linalg.norm(point_vec) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    closest_point = p1 + np.clip(t, 0, 1) * line_vec
    return np.linalg.norm(sphere_center - closest_point) <= sphere_radius


# --- 核心计算函数 ---
def get_occlusion_timeline(strategies, uav_pos_dict, target_point, time_step=0.1):
    """通用函数：计算一个或多个策略的并集遮蔽时长（简化模型）"""
    if not strategies: return 0
    
    smoke_events = []
    min_explode_time, max_end_time = float('inf'), float('-inf')
    
    for strat in strategies:
        uav_pos = uav_pos_dict[strat['uav_name']]
        v, theta_rad, t_drop, t_delay = strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']
        uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
        t_explode = t_drop + t_delay
        p_drop = uav_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        
        smoke_events.append({'t_explode': t_explode, 'p_explode': p_explode})
        min_explode_time = min(min_explode_time, t_explode)
        max_end_time = max(max_end_time, t_explode + SMOKE_DURATION)

    if min_explode_time > max_end_time: return 0

    timeline_len = int((max_end_time - min_explode_time) / time_step) + 1
    occlusion_timeline = np.zeros(timeline_len, dtype=bool)
    
    time_points = np.arange(min_explode_time, max_end_time, time_step)
    missile_positions = missile_initial_pos + missile_velocity_vector * time_points[:, np.newaxis]

    for se in smoke_events:
        relative_times = time_points - se['t_explode']
        active_mask = (relative_times >= 0) & (relative_times < SMOKE_DURATION)
        if not np.any(active_mask): continue
        
        smoke_positions = se['p_explode'] - np.array([0, 0, 3.0]) * relative_times[active_mask, np.newaxis]
        active_missile_pos = missile_positions[active_mask]

        for i in range(len(active_missile_pos)):
            if is_line_segment_intersecting_sphere(active_missile_pos[i], target_point, smoke_positions[i], SMOKE_RADIUS):
                occlusion_timeline[np.where(active_mask)[0][i]] = True
    
    return np.sum(occlusion_timeline) * time_step


# --- 阶段一：个体全局粗搜 ---
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
    
    sorted_solutions = sorted(feasible_solutions, key=lambda x: x['occlusion_time'], reverse=True)
    return sorted_solutions[:n_top]

# --- 阶段二：贪心组合 ---
def stage2_greedy_combination(individual_strats, n_teams):
    print(f"\n--- [阶段2] 开始：基于边际增益的贪心算法组合 {n_teams} 个团队 ---")
    all_strats_sorted = sorted(itertools.chain(*individual_strats.values()), key=lambda x: x['occlusion_time'], reverse=True)
    
    teams = []
    for i in range(min(n_teams, len(all_strats_sorted))):
        seed_strat = all_strats_sorted[i]
        current_uavs = [seed_strat['uav_name']]
        team_strats = [seed_strat]
        
        for _ in range(2): # 贪心选择剩下2个成员
            best_gain = -1; best_strat_to_add = None
            
            for uav_name_to_add in UAV_NAMES:
                if uav_name_to_add in current_uavs: continue
                for strat_to_add in individual_strats[uav_name_to_add]:
                    current_time = get_occlusion_timeline(team_strats + [strat_to_add], UAV_INITIAL_POS, simple_target_point)
                    if current_time > best_gain:
                        best_gain = current_time; best_strat_to_add = strat_to_add
            
            if best_strat_to_add:
                team_strats.append(best_strat_to_add)
                current_uavs.append(best_strat_to_add['uav_name'])

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

# --- 阶段三：团队协同调优 ---
def stage3_local_optimization_team(team_composition):
    params = {name: strat.copy() for name, strat in team_composition.items() if name in UAV_NAMES}
    current_max_time = team_composition.get('occlusion_time', get_occlusion_timeline(list(params.values()), UAV_INITIAL_POS, simple_target_point))
    
    for i in range(NUM_OPTIMIZATION_ROUNDS):
        last_occlusion_time = current_max_time
        for uav_name in UAV_NAMES:
            for key in ['v', 'theta_rad', 't_drop', 't_delay']:
                original_val = params[uav_name][key]
                step = 5 if key == 'v' else np.radians(10) if key == 'theta_rad' else 1.0
                search_range = np.linspace(original_val - step/2, original_val + step/2, 5)
                
                for test_val in search_range:
                    test_params_list = []
                    for name in UAV_NAMES:
                        strat_copy = params[name].copy()
                        if name == uav_name:
                            # 约束检查
                            if key == 'v' and not (UAV_V_MIN <= test_val <= UAV_V_MAX): continue
                            strat_copy[key] = test_val
                        test_params_list.append(strat_copy)
                    
                    t = get_occlusion_timeline(test_params_list, UAV_INITIAL_POS, simple_target_point)
                    if t > current_max_time:
                        current_max_time = t
                        params[uav_name][key] = test_val
        if current_max_time - last_occlusion_time < 0.1: break
    
    params['occlusion_time'] = current_max_time
    return params

# --- 阶段四：最终精度验证 ---
def stage4_final_validation_team(final_team, n_points=1000, threshold_percent=50.0):
    print(f"\n--- [阶段4] 开始：对冠军团队进行最终高精度验证 (阈值: {threshold_percent}%) ---")
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points))
    
    target_points = []
    for _ in range(int(n_points * 0.6)):
        theta, z = 2*np.pi*np.random.rand(), true_target_height*np.random.rand()
        target_points.append([true_target_base_center[0]+7*np.cos(theta), true_target_base_center[1]+7*np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = 7*np.sqrt(np.random.rand()), 2*np.pi*np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        target_points.append([true_target_base_center[0]+r*np.cos(theta), true_target_base_center[1]+r*np.sin(theta), z])
    target_points = np.array(target_points)
    
    smoke_events = []
    final_team_details = {}
    min_explode_time, max_end_time = float('inf'), float('-inf')

    for uav_name in UAV_NAMES:
        strat = final_team[uav_name]
        v, theta_rad, t_drop, t_delay = strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']
        uav_velocity = np.array([np.cos(theta_rad)*v, np.sin(theta_rad)*v, 0])
        t_explode = t_drop + t_delay
        p_drop = UAV_INITIAL_POS[uav_name] + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity*t_delay + np.array([0,0,-0.5*GRAVITY*t_delay**2])
        
        smoke_events.append({'t_explode': t_explode, 'p_explode': p_explode})
        final_team_details[uav_name] = {**strat, 'p_drop': p_drop, 'p_explode': p_explode, 't_explode': t_explode}
        min_explode_time = min(min_explode_time, t_explode)
        max_end_time = max(max_end_time, t_explode + SMOKE_DURATION)

    total_occluded_time = 0.0
    time_step = 0.01
    for t in np.arange(min_explode_time, max_end_time, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        is_occluded_this_step = False
        for se in smoke_events:
            if se['t_explode'] <= t < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -3.0 * (t - se['t_explode'])])
                occluded_lines_count = sum(1 for tp in target_points if is_line_segment_intersecting_sphere(missile_pos, tp, smoke_pos, SMOKE_RADIUS))
                if occluded_lines_count >= required_occluded_count:
                    is_occluded_this_step = True
                    break
        if is_occluded_this_step: total_occluded_time += time_step
    
    return total_occluded_time, final_team_details

# --- 主程序 ---
if __name__ == "__main__":
    start_total_time = time.time()
    
    print("="*60)
    print("        第四问：三机协同干扰问题求解器启动")
    print("="*60)
    
    individual_strategies = {name: stage1_individual_coarse_search(name, UAV_INITIAL_POS[name], NUM_INDIVIDUAL_STRATEGIES) for name in UAV_NAMES}
    initial_teams = stage2_greedy_combination(individual_strategies, NUM_TEAMS_TO_FORM)

    if not initial_teams:
        print("错误：阶段二未能组合出任何团队。")
    else:
        print(f"\n--- [阶段3] 开始：对 {len(initial_teams)} 个团队进行局部协同调优 ---")
        optimized_teams = []
        for i, team in enumerate(initial_teams):
            print(f"  - 正在优化团队 {i+1}/{len(initial_teams)} (初始时长: {team['occlusion_time']:.2f}s)...", end='')
            optimized_team = stage3_local_optimization_team(team)
            optimized_teams.append(optimized_team)
            print(f" 优化后时长: {optimized_team['occlusion_time']:.4f}s")
        
        champion_team = max(optimized_teams, key=lambda x: x['occlusion_time'])
        
        final_time, final_details = stage4_final_validation_team(champion_team, threshold_percent=OCCLUSION_THRESHOLD_PERCENT)
        
        print("\n" + "="*60)
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