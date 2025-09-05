import numpy as np
import time

# --- 0. 基础设置与可调参数 ---
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0
DROP_INTERVAL = 1.0 # 最小投放间隔

# 多起点优化的起点数量
NUM_INITIAL_STARTS = 50

# 最终验证的遮蔽阈值
OCCLUSION_THRESHOLD_PERCENT = 70.0

# 初始位置等 (与之前相同)
uav_initial_pos = np.array([17800, 0, 1800], dtype=float)
missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * 300.0
missile_total_time = np.linalg.norm(false_target_pos - missile_initial_pos) / 300.0

# 步骤1 搜索参数 (与之前相同)
SEARCH_EXPLODE_TIMES_RANGE = (missile_total_time * 0, missile_total_time * 0.9)
SEARCH_EXPLODE_TIMES_STEPS = 2000
SEARCH_DELAYS_RANGE = (0, 5.0)
SEARCH_DELAYS_STEPS = 150
SEARCH_LOS_RATIO_RANGE = (0, 0.9)
SEARCH_LOS_RATIO_STEPS = 900

# --- 辅助函数 (与之前版本完全相同) ---
# ... (此处省略 check_reachability, is_line_segment_intersecting_sphere, calculate_simple_occlusion_time 等函数)
def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius):
    if np.any(np.isnan(sphere_center)): return False
    line_vec, point_vec = p2 - p1, sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.linalg.norm(point_vec) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    closest_point = p1 + np.clip(t, 0, 1) * line_vec
    return np.linalg.norm(sphere_center - closest_point) <= sphere_radius

def calculate_simple_occlusion_time(uav_vel, uav_dir_rad, t_drop, t_delay):
    uav_velocity = np.array([np.cos(uav_dir_rad) * uav_vel, np.sin(uav_dir_rad) * uav_vel, 0])
    t_explode = t_drop + t_delay
    p_drop = uav_initial_pos + uav_velocity * t_drop
    p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
    occluded_time = 0
    time_step = 0.1
    for t in np.arange(t_explode, t_explode + SMOKE_DURATION, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        smoke_pos = p_explode + np.array([0, 0, -3.0 * (t - t_explode)])
        if is_line_segment_intersecting_sphere(missile_pos, simple_target_point, smoke_pos, SMOKE_RADIUS):
            occluded_time += time_step
    return occluded_time

def check_reachability(t_explode, t_delay, los_ratio):
    if t_explode <= t_delay: return False, None, None
    missile_pos_at_explode = missile_initial_pos + missile_velocity_vector * t_explode
    los_vector = simple_target_point - missile_pos_at_explode
    ideal_explode_pos = missile_pos_at_explode + los_ratio * los_vector
    t_drop = t_explode - t_delay
    if t_drop <= 0: return False, None, None
    required_uav_velocity_3d = (ideal_explode_pos - uav_initial_pos - np.array([0, 0, -0.5 * GRAVITY * t_delay**2])) / t_explode
    required_uav_velocity_xy = required_uav_velocity_3d[:2]
    required_speed = np.linalg.norm(required_uav_velocity_xy)
    if UAV_V_MIN <= required_speed <= UAV_V_MAX:
        uav_dir_rad = np.arctan2(required_uav_velocity_xy[1], required_uav_velocity_xy[0])
        return True, required_speed, uav_dir_rad
    else:
        return False, None, None

def calculate_simple_total_occlusion_time(params):
    v, theta_rad = params['v'], params['theta_rad']
    drops = [params['drop1'], params['drop2'], params['drop3']]
    uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
    smoke_trajectories = []
    for t_drop, t_delay in drops:
        t_explode = t_drop + t_delay
        p_drop = uav_initial_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        smoke_trajectories.append({'t_explode': t_explode, 'p_explode': p_explode})
    total_occluded_time = 0
    time_step = 0.01
    min_explode_time = min(st['t_explode'] for st in smoke_trajectories)
    max_end_time = max(st['t_explode'] for st in smoke_trajectories) + SMOKE_DURATION
    for t in np.arange(min_explode_time, max_end_time, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        is_occluded_this_step = False
        for st in smoke_trajectories:
            if st['t_explode'] <= t < st['t_explode'] + SMOKE_DURATION:
                smoke_pos = st['p_explode'] + np.array([0, 0, -3.0 * (t - st['t_explode'])])
                if is_line_segment_intersecting_sphere(missile_pos, simple_target_point, smoke_pos, SMOKE_RADIUS):
                    is_occluded_this_step = True
                    break
        if is_occluded_this_step: total_occluded_time += time_step
    return total_occluded_time

# --- 步骤 1 (不变) ---
def step1_find_top_N_windows(n):
    # ... (与上一版完全相同)
    print(f"--- [步骤 1] 开始：搜索 Top {n} 个最优初始解 ---")
    feasible_solutions = []
    search_explode_times = np.linspace(*SEARCH_EXPLODE_TIMES_RANGE, SEARCH_EXPLODE_TIMES_STEPS)
    search_delays = np.linspace(*SEARCH_DELAYS_RANGE, SEARCH_DELAYS_STEPS)
    search_los_ratios = np.linspace(*SEARCH_LOS_RATIO_RANGE, SEARCH_LOS_RATIO_STEPS)
    for t_explode in search_explode_times:
        for t_delay in search_delays:
            for los_ratio in search_los_ratios:
                is_feasible, req_speed, req_dir_rad = check_reachability(t_explode, t_delay, los_ratio)
                if not is_feasible: continue
                t_drop = t_explode - t_delay
                current_occlusion_time = calculate_simple_occlusion_time(req_speed, req_dir_rad, t_drop, t_delay)
                if current_occlusion_time > 0: feasible_solutions.append({'v': req_speed, 'theta_rad': req_dir_rad, 't_drop': t_drop, 't_delay': t_delay, 'los_ratio': los_ratio, 'occlusion_time': current_occlusion_time})
    if not feasible_solutions: return []
    sorted_solutions = sorted(feasible_solutions, key=lambda x: x['occlusion_time'], reverse=True)
    top_n_solutions = []
    for sol in sorted_solutions:
        if not any(np.linalg.norm([sol['v'] - ts['v'], (sol['t_drop']-ts['t_drop'])*10]) < 5.0 for ts in top_n_solutions):
            top_n_solutions.append(sol)
        if len(top_n_solutions) >= n: break
    print(f"步骤1完成。从{len(feasible_solutions)}个可行解中筛选出 {len(top_n_solutions)} 个高质量初始解。")
    return top_n_solutions

# --- 步骤 2 (应用新的初始化策略) ---
def step2_local_optimization_multi(initial_params):
    print("\n--- [步骤 2] 开始：对3枚烟幕弹进行8维协同优化 (采用'中心饱和'初始策略) ---")
    
    # --- 核心修改在此处 ---
    t_drop_center = initial_params['t_drop']
    base_delay = initial_params['t_delay']
    t_drop_early = t_drop_center - DROP_INTERVAL - 0.2
    t_drop_late = t_drop_center + DROP_INTERVAL + 0.2
    if t_drop_early < 0.1:
        t_drop_early = 0.1
        t_drop_center = t_drop_early + DROP_INTERVAL
        t_drop_late = t_drop_center + DROP_INTERVAL
    
    params = {
        'v': initial_params['v'],
        'theta_rad': initial_params['theta_rad'],
        'drop1': (t_drop_early, base_delay),
        'drop2': (t_drop_center, base_delay),
        'drop3': (t_drop_late, base_delay),
    }
    # --- 初始化结束 ---
    
    current_max_time = calculate_simple_total_occlusion_time(params)
    
    # 后续的优化循环与之前完全相同
    for i in range(5):
        last_occlusion_time = current_max_time
        for v_test in np.linspace(max(UAV_V_MIN, params['v']-5), min(UAV_V_MAX, params['v']+5), 11):
            test_params = params.copy(); test_params['v'] = v_test
            t = calculate_simple_total_occlusion_time(test_params)
            if t > current_max_time: current_max_time, params['v'] = t, v_test
        for theta_test in np.linspace(params['theta_rad']-np.radians(10), params['theta_rad']+np.radians(10), 11):
            test_params = params.copy(); test_params['theta_rad'] = theta_test
            t = calculate_simple_total_occlusion_time(test_params)
            if t > current_max_time: current_max_time, params['theta_rad'] = t, theta_test
        for j in range(1, 4):
            key = f'drop{j}'; t_drop_base, t_delay_base = params[key]
            min_t_drop = params[f'drop{j-1}'][0] + DROP_INTERVAL if j > 1 else 0.1
            for t_drop_test in np.linspace(max(min_t_drop, t_drop_base-1), t_drop_base+1, 11):
                test_params = params.copy(); test_params[key] = (t_drop_test, t_delay_base)
                if j < 3 and t_drop_test + DROP_INTERVAL > params[f'drop{j+1}'][0]: test_params[f'drop{j+1}'] = (t_drop_test + DROP_INTERVAL, params[f'drop{j+1}'][1])
                t = calculate_simple_total_occlusion_time(test_params)
                if t > current_max_time: current_max_time, params = t, test_params
            t_drop_base, t_delay_base = params[key]
            for t_delay_test in np.linspace(max(0.1, t_delay_base-1), t_delay_base+1, 11):
                test_params = params.copy(); test_params[key] = (t_drop_base, t_delay_test)
                t = calculate_simple_total_occlusion_time(test_params)
                if t > current_max_time: current_max_time, params[key] = t, (t_drop_base, t_delay_test)
        if current_max_time - last_occlusion_time < 0.1: break
    params['occlusion_time'] = current_max_time
    return params

# --- 步骤 3 (不变) ---
def step3_final_validation_multi(final_params, n_points=1000, threshold_percent=70.0):
    # ... (与上一版完全相同，此处省略)
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points))
    target_points = []
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), true_target_height * np.random.rand()
        target_points.append([true_target_base_center[0] + 7 * np.cos(theta), true_target_base_center[1] + 7 * np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = 7 * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        target_points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), z])
    target_points = np.array(target_points)
    v, theta_rad = final_params['v'], final_params['theta_rad']
    uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
    smoke_events = []
    for i in range(1, 4):
        t_drop, t_delay = final_params[f'drop{i}']
        t_explode = t_drop + t_delay
        p_drop = uav_initial_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        smoke_events.append({'t_drop': t_drop, 't_delay': t_delay, 't_explode': t_explode, 'p_drop': p_drop, 'p_explode': p_explode})
    total_occluded_time = 0.0
    time_step = 0.001
    min_explode_time = min(se['t_explode'] for se in smoke_events)
    max_end_time = max(se['t_explode'] for se in smoke_events) + SMOKE_DURATION
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
    return total_occluded_time, final_params, smoke_events

# --- 主程序 (不变) ---
if __name__ == "__main__":
    # ... (与上一版完全相同，此处省略)
    start_total_time = time.time()
    initial_solutions_list = step1_find_top_N_windows(n=NUM_INITIAL_STARTS)
    if not initial_solutions_list:
        print("错误：第一步未能找到任何可行的初始解。")
    else:
        local_optima_list = []
        for i, initial_sol in enumerate(initial_solutions_list):
            print(f"\n--- 对第 {i+1}/{len(initial_solutions_list)} 个初始解进行局部优化 (初始时长: {initial_sol['occlusion_time']:.2f}s) ---")
            optimized_solution = step2_local_optimization_multi(initial_sol)
            local_optima_list.append(optimized_solution)
            print(f"--- 优化完成，得到的局部最优时长: {optimized_solution['occlusion_time']:.4f}s ---")
        best_overall_solution = max(local_optima_list, key=lambda x: x['occlusion_time'])
        print("\n--- 对全局最优候选解进行最终高精度验证 ---")
        final_time, final_params, smoke_events = step3_final_validation_multi(best_overall_solution, threshold_percent=OCCLUSION_THRESHOLD_PERCENT)
        print("\n" + "="*60)
        print("        第三问：最优干扰策略及最终结果 (多起点优化)")
        print("="*60)
        print(f"\n搜索概况: 从 {len(initial_solutions_list)} 个高质量起点出发，找到的最佳策略如下。")
        print("\n--- 决策变量 (UAV统一飞行策略) ---")
        print(f"  无人机飞行速度 (v)      : {final_params['v']:.4f} m/s")
        print(f"  无人机飞行方向 (θ)      : {np.degrees(final_params['theta_rad']):.4f} 度")
        for i, se in enumerate(smoke_events):
            print(f"\n--- 第 {i+1} 枚烟幕弹 ---")
            print(f"  投放时间 (t_drop)       : {se['t_drop']:.4f} 秒")
            print(f"  引信延迟时间 (t_delay)  : {se['t_delay']:.4f} 秒")
            print(f"  投放点坐标 (P_drop)   : ({se['p_drop'][0]:.2f}, {se['p_drop'][1]:.2f}, {se['p_drop'][2]:.2f})")
            print(f"  起爆点坐标 (P_explode): ({se['p_explode'][0]:.2f}, {se['p_explode'][1]:.2f}, {se['p_explode'][2]:.2f})")
            print(f"  起爆时间 (t_explode)  : {se['t_explode']:.4f} 秒")
        print("\n--- 最终性能评估 ---")
        print(f"  遮蔽有效性阈值          : {OCCLUSION_THRESHOLD_PERCENT:.1f}%")
        print(f"  最终高精度总有效遮蔽时长: {final_time:.4f} 秒")
        print("="*60)
    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")