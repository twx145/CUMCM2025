import numpy as np
import time

# --- 0. 基础设置与可调参数 ---
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0

# !!! 1. 新增：可调的有效遮蔽阈值 !!!
# 在最终验证阶段，至少需要遮蔽 N% 的视线才算作有效遮蔽。
# 例如：50.0 表示至少要遮住一半的采样点。
#        0.1 表示只要遮住任何一个点就算。
OCCLUSION_THRESHOLD_PERCENT = 70.0

# 初始位置
uav_initial_pos = np.array([17800, 0, 1800], dtype=float)
missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])

# 导弹轨迹计算
missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * 300.0
missile_total_time = np.linalg.norm(false_target_pos - missile_initial_pos) / 300.0

# --- 步骤1 搜索参数 ---
SEARCH_EXPLODE_TIMES_RANGE = (missile_total_time * 0, missile_total_time * 0.9)
SEARCH_EXPLODE_TIMES_STEPS = 200
SEARCH_DELAYS_RANGE = (0, 8.0)
SEARCH_DELAYS_STEPS = 150
SEARCH_LOS_RATIO_RANGE = (0, 0.9)
SEARCH_LOS_RATIO_STEPS = 90

# --- 辅助函数 (与之前版本相同) ---
# ... (此处省略 check_reachability, is_line_segment_intersecting_sphere, calculate_simple_occlusion_time 函数，它们与上一版完全相同)
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


# --- 步骤 1 & 2 (与之前版本相同) ---
# ... (此处省略 step1_find_optimal_window 和 step2_local_optimization 函数，它们与上一版完全相同)
def step1_find_optimal_window():
    print("--- [步骤 1] 开始：基于前置可行性判断的剪枝搜索 ---")
    best_params = {}
    max_occlusion_time = -1
    search_explode_times = np.linspace(*SEARCH_EXPLODE_TIMES_RANGE, SEARCH_EXPLODE_TIMES_STEPS)
    search_delays = np.linspace(*SEARCH_DELAYS_RANGE, SEARCH_DELAYS_STEPS)
    search_los_ratios = np.linspace(*SEARCH_LOS_RATIO_RANGE, SEARCH_LOS_RATIO_STEPS)
    feasible_count = 0
    for t_explode in search_explode_times:
        for t_delay in search_delays:
            for los_ratio in search_los_ratios:
                is_feasible, req_speed, req_dir_rad = check_reachability(t_explode, t_delay, los_ratio)
                if not is_feasible: continue
                feasible_count += 1
                t_drop = t_explode - t_delay
                current_occlusion_time = calculate_simple_occlusion_time(req_speed, req_dir_rad, t_drop, t_delay)
                if current_occlusion_time > max_occlusion_time:
                    max_occlusion_time = current_occlusion_time
                    best_params = {'v': req_speed, 'theta_rad': req_dir_rad, 't_drop': t_drop, 't_delay': t_delay, 'los_ratio': los_ratio, 'occlusion_time': max_occlusion_time}
    print(f"步骤1完成。找到初始解，可行方案数: {feasible_count}")
    return best_params

def step2_local_optimization(initial_params):
    print("\n--- [步骤 2] 开始：使用坐标上升法进行局部精细搜索 ---")
    params = initial_params.copy()
    for i in range(5):
        last_occlusion_time = params['occlusion_time']
        for v_test in np.linspace(max(70, params['v']-5), min(140, params['v']+5), 11):
            t = calculate_simple_occlusion_time(v_test, params['theta_rad'], params['t_drop'], params['t_delay'])
            if t > params['occlusion_time']: params['occlusion_time'], params['v'] = t, v_test
        for theta_test in np.linspace(params['theta_rad']-np.radians(10), params['theta_rad']+np.radians(10), 11):
            t = calculate_simple_occlusion_time(params['v'], theta_test, params['t_drop'], params['t_delay'])
            if t > params['occlusion_time']: params['occlusion_time'], params['theta_rad'] = t, theta_test
        for t_drop_test in np.linspace(max(0.1, params['t_drop']-1), params['t_drop']+1, 11):
            t = calculate_simple_occlusion_time(params['v'], params['theta_rad'], t_drop_test, params['t_delay'])
            if t > params['occlusion_time']: params['occlusion_time'], params['t_drop'] = t, t_drop_test
        for t_delay_test in np.linspace(max(0.1, params['t_delay']-1), params['t_delay']+1, 11):
            t = calculate_simple_occlusion_time(params['v'], params['theta_rad'], params['t_drop'], t_delay_test)
            if t > params['occlusion_time']: params['occlusion_time'], params['t_delay'] = t, t_delay_test
        if params['occlusion_time'] - last_occlusion_time < 0.01: break
    print(f"步骤2完成。")
    return params


# --- 步骤 3: 最终精度验证 (集成阈值判断和详细打印) ---
def step3_final_validation(final_params, n_points=1000, threshold_percent=70.0):
    print(f"\n--- [步骤 3] 开始：使用{n_points}个目标点和{threshold_percent}%的遮蔽阈值进行最终验证 ---")
    
    # 计算需要遮蔽的视线数量
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points))
    print(f"有效遮蔽条件: 至少需要遮蔽 {required_occluded_count} / {n_points} 条视线。")

    # 生成目标表面采样点
    points = []
    # (为简洁省略具体生成代码，与上一版相同)
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), true_target_height * np.random.rand()
        points.append([true_target_base_center[0] + 7 * np.cos(theta), true_target_base_center[1] + 7 * np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = 7 * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), z])
    target_points = np.array(points)
    
    # 提取最优参数
    v, theta_rad, t_drop, t_delay = final_params['v'], final_params['theta_rad'], final_params['t_drop'], final_params['t_delay']
    
    # 计算最终的轨迹和关键点
    uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
    t_explode = t_drop + t_delay
    p_drop = uav_initial_pos + uav_velocity * t_drop
    p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])

    # 高精度计算
    total_occluded_time = 0.0
    time_step = 0.001
    for t in np.arange(t_explode, t_explode + SMOKE_DURATION, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        smoke_pos = p_explode + np.array([0, 0, -3.0 * (t - t_explode)])
        
        # --- 核心修改：计数并与阈值比较 ---
        occluded_lines_count = sum(1 for tp in target_points if is_line_segment_intersecting_sphere(missile_pos, tp, smoke_pos, SMOKE_RADIUS))
        
        if occluded_lines_count >= required_occluded_count:
            total_occluded_time += time_step
            
    # --- 2. 新增：全面的最终信息打印 ---
    print("\n" + "="*60)
    print("        第二问：最优干扰策略及最终结果")
    print("="*60)
    print("\n--- 决策变量 (UAV飞行与投放策略) ---")
    print(f"  无人机飞行速度 (v)      : {v:.4f} m/s")
    print(f"  无人机飞行方向 (θ)      : {np.degrees(theta_rad):.4f} 度")
    print(f"  无人机飞行时间 (至投放) : {t_drop:.4f} 秒")
    print(f"  烟幕弹投放时间 (t_drop) : {t_drop:.4f} 秒 (自任务开始)")
    print(f"  引信延迟时间 (t_delay)    : {t_delay:.4f} 秒")
    
    print("\n--- 关键事件节点 ---")
    print(f"  投放点坐标 (P_drop)     : ({p_drop[0]:.2f}, {p_drop[1]:.2f}, {p_drop[2]:.2f})")
    print(f"  起爆点坐标 (P_explode)  : ({p_explode[0]:.2f}, {p_explode[1]:.2f}, {p_explode[2]:.2f})")
    print(f"  起爆时间 (t_explode)    : {t_explode:.4f} 秒 (自任务开始)")

    print("\n--- 最终性能评估 ---")
    print(f"  遮蔽有效性阈值          : {threshold_percent:.1f}%")
    print(f"  最终高精度有效遮蔽时长  : {total_occluded_time:.4f} 秒")
    print("="*60)
    
    return total_occluded_time

# --- 主程序 ---
if __name__ == "__main__":
    start_total_time = time.time()
    
    initial_solution = step1_find_optimal_window()
    if not initial_solution:
        print("错误：第一步未能找到任何可行的初始解，请检查搜索范围或约束条件。")
    else:
        optimized_solution = step2_local_optimization(initial_solution)
        # 将全局设定的阈值传入最终验证函数
        final_result = step3_final_validation(optimized_solution, threshold_percent=OCCLUSION_THRESHOLD_PERCENT)
    
    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")