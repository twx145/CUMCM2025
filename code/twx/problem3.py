import numpy as np
import time

# --- 0. 基础设置与可调参数 ---
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0
DROP_INTERVAL = 1.0 # 最小投放间隔

# 可调的有效遮蔽阈值
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
SEARCH_EXPLODE_TIMES_STEPS = 500
SEARCH_DELAYS_RANGE = (0, 8.0)
SEARCH_DELAYS_STEPS = 500
SEARCH_LOS_RATIO_RANGE = (0, 0.9)
SEARCH_LOS_RATIO_STEPS = 500

# --- 辅助函数 ---
# is_line_segment_intersecting_sphere, check_reachability, calculate_simple_occlusion_time (单枚) 保持不变
def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius):
    if np.any(np.isnan(sphere_center)): return False
    line_vec, point_vec = p2 - p1, sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.linalg.norm(point_vec) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    closest_point = p1 + np.clip(t, 0, 1) * line_vec
    return np.linalg.norm(sphere_center - closest_point) <= sphere_radius

def calculate_simple_occlusion_time(uav_vel, uav_dir_rad, t_drop, t_delay):
    # ... (与上一版完全相同)
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
    # ... (与上一版完全相同)
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


# --- 新增：计算3枚弹总遮蔽时间的目标函数 ---
def calculate_simple_total_occlusion_time(params):
    v, theta_rad = params['v'], params['theta_rad']
    drops = [params['drop1'], params['drop2'], params['drop3']]
    
    uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
    
    # 计算每枚烟幕弹的轨迹
    smoke_trajectories = []
    for t_drop, t_delay in drops:
        t_explode = t_drop + t_delay
        p_drop = uav_initial_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        smoke_trajectories.append({'t_explode': t_explode, 'p_explode': p_explode})
        
    # 计算并集遮蔽时间
    total_occluded_time = 0
    time_step = 0.01
    # 确定模拟的时间范围，以包含所有烟幕云
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
                    break # 只要有一个遮蔽，该时间点就算遮蔽
        if is_occluded_this_step:
            total_occluded_time += time_step
            
    return total_occluded_time

# --- 步骤 1 (不变) & 步骤 2 (升级为8维优化) ---
def step1_find_optimal_window():
    # ... (与上一版完全相同)
    print("--- [步骤 1] 开始：为基础航线寻找最优初始解 (基于单枚烟幕弹) ---")
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

def step2_local_optimization_multi(initial_params):
    print("\n--- [步骤 2] 开始：对3枚烟幕弹进行8维协同优化 ---")
    
    # 初始化8个参数
    params = {
        'v': initial_params['v'],
        'theta_rad': initial_params['theta_rad'],
        'drop1': (initial_params['t_drop'], initial_params['t_delay']),
    }
    params['drop2'] = (params['drop1'][0] + 1.2, params['drop1'][1])
    params['drop3'] = (params['drop2'][0] + 1.2, params['drop2'][1])
    
    current_max_time = calculate_simple_total_occlusion_time(params)
    
    for i in range(5): # 迭代5轮
        print(f"迭代轮次 {i+1}/5...")
        last_occlusion_time = current_max_time
        
        # 依次优化8个变量
        # 优化 v
        for v_test in np.linspace(max(UAV_V_MIN, params['v']-5), min(UAV_V_MAX, params['v']+5), 11):
            test_params = params.copy(); test_params['v'] = v_test
            t = calculate_simple_total_occlusion_time(test_params)
            if t > current_max_time: current_max_time, params['v'] = t, v_test
        
        # 优化 theta_rad
        for theta_test in np.linspace(params['theta_rad']-np.radians(10), params['theta_rad']+np.radians(10), 11):
            test_params = params.copy(); test_params['theta_rad'] = theta_test
            t = calculate_simple_total_occlusion_time(test_params)
            if t > current_max_time: current_max_time, params['theta_rad'] = t, theta_test

        # 优化三枚弹的 t_drop 和 t_delay
        for j in range(1, 4):
            key = f'drop{j}'
            # 优化 t_drop
            t_drop_base, t_delay_base = params[key]
            # 必须满足间隔约束
            min_t_drop = params[f'drop{j-1}'][0] + DROP_INTERVAL if j > 1 else 0.1
            for t_drop_test in np.linspace(max(min_t_drop, t_drop_base-1), t_drop_base+1, 11):
                test_params = params.copy(); test_params[key] = (t_drop_test, t_delay_base)
                # 如果影响了后续投放，进行联动修改 (简单策略)
                if j < 3 and t_drop_test + DROP_INTERVAL > params[f'drop{j+1}'][0]:
                    test_params[f'drop{j+1}'] = (t_drop_test + DROP_INTERVAL, params[f'drop{j+1}'][1])
                
                t = calculate_simple_total_occlusion_time(test_params)
                if t > current_max_time: current_max_time, params = t, test_params
            
            # 优化 t_delay
            t_drop_base, t_delay_base = params[key]
            for t_delay_test in np.linspace(max(0.1, t_delay_base-1), t_delay_base+1, 11):
                test_params = params.copy(); test_params[key] = (t_drop_base, t_delay_test)
                t = calculate_simple_total_occlusion_time(test_params)
                if t > current_max_time: current_max_time, params[key] = t, (t_drop_base, t_delay_test)

        if current_max_time - last_occlusion_time < 0.1: # 放宽收敛阈值
            print("总遮蔽时长提升小于0.1s，算法收敛。")
            break
            
    params['occlusion_time'] = current_max_time
    print(f"步骤2完成。预估总遮蔽时长 (简化模型): {params['occlusion_time']:.4f} s")
    return params

# --- 步骤 3: 最终精度验证 (升级版) ---
def step3_final_validation_multi(final_params, n_points=1000, threshold_percent=50.0):
    print(f"\n--- [步骤 3] 开始：对3枚烟幕弹进行最终精度验证 (阈值: {threshold_percent}%) ---")
    
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points))
    print(f"有效遮蔽条件: 至少需要遮蔽 {required_occluded_count} / {n_points} 条视线。")
    target_points = [] # 此处省略生成点的代码
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), true_target_height * np.random.rand()
        target_points.append([true_target_base_center[0] + 7 * np.cos(theta), true_target_base_center[1] + 7 * np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = 7 * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        target_points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), z])
    target_points = np.array(target_points)
    
    # 计算所有烟幕弹的轨迹
    v, theta_rad = final_params['v'], final_params['theta_rad']
    uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
    
    smoke_events = []
    for i in range(1, 4):
        t_drop, t_delay = final_params[f'drop{i}']
        t_explode = t_drop + t_delay
        p_drop = uav_initial_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        smoke_events.append({'t_drop': t_drop, 't_delay': t_delay, 't_explode': t_explode, 'p_drop': p_drop, 'p_explode': p_explode})

    # 高精度计算总遮蔽时长
    total_occluded_time = 0.0
    time_step = 0.01
    min_explode_time = min(se['t_explode'] for se in smoke_events)
    max_end_time = max(se['t_explode'] for se in smoke_events) + SMOKE_DURATION
    
    for t in np.arange(min_explode_time, max_end_time, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        
        # 检查任何一个云团是否造成了满足阈值的遮蔽
        is_occluded_this_step = False
        for se in smoke_events:
            if se['t_explode'] <= t < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -3.0 * (t - se['t_explode'])])
                occluded_lines_count = sum(1 for tp in target_points if is_line_segment_intersecting_sphere(missile_pos, tp, smoke_pos, SMOKE_RADIUS))
                if occluded_lines_count >= required_occluded_count:
                    is_occluded_this_step = True
                    break # 只要有一个满足条件，该时间点就有效
        
        if is_occluded_this_step:
            total_occluded_time += time_step
            
    # 打印最终详细报告
    print("\n" + "="*60)
    print("        第三问：最优干扰策略及最终结果")
    print("="*60)
    print("\n--- 决策变量 (UAV统一飞行策略) ---")
    print(f"  无人机飞行速度 (v)      : {v:.4f} m/s")
    print(f"  无人机飞行方向 (θ)      : {np.degrees(theta_rad):.4f} 度")
    
    for i, se in enumerate(smoke_events):
        print(f"\n--- 第 {i+1} 枚烟幕弹 ---")
        print(f"  投放时间 (t_drop)       : {se['t_drop']:.4f} 秒")
        print(f"  引信延迟时间 (t_delay)  : {se['t_delay']:.4f} 秒")
        print(f"  投放点坐标 (P_drop)   : ({se['p_drop'][0]:.2f}, {se['p_drop'][1]:.2f}, {se['p_drop'][2]:.2f})")
        print(f"  起爆点坐标 (P_explode): ({se['p_explode'][0]:.2f}, {se['p_explode'][1]:.2f}, {se['p_explode'][2]:.2f})")
        print(f"  起爆时间 (t_explode)  : {se['t_explode']:.4f} 秒")

    print("\n--- 最终性能评估 ---")
    print(f"  遮蔽有效性阈值          : {threshold_percent:.1f}%")
    print(f"  最终高精度总有效遮蔽时长: {total_occluded_time:.4f} 秒")
    print("="*60)
    
    return total_occluded_time

# --- 主程序 ---
if __name__ == "__main__":
    start_total_time = time.time()
    
    initial_solution = step1_find_optimal_window()
    if not initial_solution:
        print("错误：第一步未能找到任何可行的初始解。")
    else:
        # 调用为多弹优化的第二步
        optimized_solution_multi = step2_local_optimization_multi(initial_solution)
        # 调用为多弹验证的第三步
        final_result = step3_final_validation_multi(optimized_solution_multi, threshold_percent=OCCLUSION_THRESHOLD_PERCENT)
    
    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")