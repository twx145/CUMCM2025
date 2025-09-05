import numpy as np
import time

# --- 0. 基础设置与物理常量 ---
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0

# 初始位置
uav_initial_pos = np.array([17800, 0, 1800], dtype=float)
missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
# 简化模型的目标点 (步骤1和2使用)
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])

# 导弹轨迹计算
missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * 300.0
missile_total_time = np.linalg.norm(false_target_pos - missile_initial_pos) / 300.0

# --- 步骤1 搜索参数 ---
# 通过调整这里的参数可以改变第一步粗搜索的范围和精度
SEARCH_EXPLODE_TIMES_RANGE = (missile_total_time * 0, missile_total_time * 0.2)
SEARCH_EXPLODE_TIMES_STEPS = 500
SEARCH_DELAYS_RANGE = (2.0, 8.0)
SEARCH_DELAYS_STEPS = 150
SEARCH_LOS_RATIO_RANGE = (0.01, 0.9) # 0.1=靠近导弹, 0.9=靠近目标
SEARCH_LOS_RATIO_STEPS = 90          # 从0.1到0.9，步长0.1

# --- 辅助函数 (与之前版本相同) ---
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
    v_drop = uav_velocity
    p_explode = p_drop + v_drop * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])

    occluded_time = 0
    time_step = 0.1
    for t in np.arange(t_explode, t_explode + SMOKE_DURATION, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        smoke_pos = p_explode + np.array([0, 0, -3.0 * (t - t_explode)])
        if is_line_segment_intersecting_sphere(missile_pos, simple_target_point, smoke_pos, SMOKE_RADIUS):
            occluded_time += time_step
    return occluded_time

# --- 步骤 1: 全局粗搜索 (升级版) ---
def step1_find_optimal_window():
    print("--- [步骤 1] 开始：全局粗搜索以寻找最优拦截窗口 ---")
    best_params = {}
    max_occlusion_time = -1
    
    # 离散化搜索空间
    search_explode_times = np.linspace(SEARCH_EXPLODE_TIMES_RANGE[0], SEARCH_EXPLODE_TIMES_RANGE[1], SEARCH_EXPLODE_TIMES_STEPS)
    search_delays = np.linspace(SEARCH_DELAYS_RANGE[0], SEARCH_DELAYS_RANGE[1], SEARCH_DELAYS_STEPS)
    search_los_ratios = np.linspace(SEARCH_LOS_RATIO_RANGE[0], SEARCH_LOS_RATIO_RANGE[1], SEARCH_LOS_RATIO_STEPS)
    
    total_iterations = len(search_explode_times) * len(search_delays) * len(search_los_ratios)
    print(f"粗搜索总迭代次数: {total_iterations}")

    for t_explode in search_explode_times:
        missile_pos_at_explode = missile_initial_pos + missile_velocity_vector * t_explode
        los_vector = simple_target_point - missile_pos_at_explode
            
        for t_delay in search_delays:
            if t_explode <= t_delay: continue
            
            for los_ratio in search_los_ratios:
                ideal_explode_pos = missile_pos_at_explode + los_ratio * los_vector
                t_drop = t_explode - t_delay
                required_uav_velocity_3d = (ideal_explode_pos - uav_initial_pos - np.array([0, 0, -0.5 * GRAVITY * t_delay**2])) / t_explode
                required_uav_velocity_3d[2] = 0
                required_speed = np.linalg.norm(required_uav_velocity_3d)

                if 70 <= required_speed <= 140:
                    uav_dir_rad = np.arctan2(required_uav_velocity_3d[1], required_uav_velocity_3d[0])
                    current_occlusion_time = calculate_simple_occlusion_time(required_speed, uav_dir_rad, t_drop, t_delay)
                    
                    if current_occlusion_time > max_occlusion_time:
                        max_occlusion_time = current_occlusion_time
                        best_params = {
                            'v': required_speed,
                            'theta_rad': uav_dir_rad,
                            't_drop': t_drop,
                            't_delay': t_delay,
                            'los_ratio': los_ratio,
                            'occlusion_time': max_occlusion_time
                        }
    
    if not best_params:
        print("步骤1失败：在当前搜索范围内，未能找到任何可行的无人机策略。")
        print("建议：请尝试进一步扩大步骤1的搜索参数范围，特别是 SEARCH_DELAYS_RANGE。")
        return None  # 返回 None 表示失败
    
    print(f"步骤1完成。找到的最佳初始解：")
    print(f"  速度: {best_params['v']:.2f} m/s, 方向: {np.degrees(best_params['theta_rad']):.2f} 度")
    print(f"  投放时间: {best_params['t_drop']:.2f} s, 延迟引爆: {best_params['t_delay']:.2f} s")
    print(f"  最优视线比例: {best_params['los_ratio']:.2f}")
    print(f"  预估遮蔽时长 (简化模型): {best_params['occlusion_time']:.2f} s")
    return best_params

# --- 步骤 2: 局部精搜索 (与之前版本相同) ---
def step2_local_optimization(initial_params):
    print("\n--- [步骤 2] 开始：使用坐标上升法进行局部精细搜索 ---")
    
    params = initial_params.copy()
    
    for i in range(5):
        print(f"迭代轮次 {i+1}/5...")
        last_occlusion_time = params['occlusion_time']
        
        # 优化 v, theta, t_drop, t_delay
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

        if params['occlusion_time'] - last_occlusion_time < 0.01:
            print("遮蔽时长提升小于0.01s，算法收敛。")
            break
            
    print(f"步骤2完成。优化后的解：")
    print(f"  速度: {params['v']:.4f} m/s, 方向: {np.degrees(params['theta_rad']):.4f} 度")
    print(f"  投放时间: {params['t_drop']:.4f} s, 延迟引爆: {params['t_delay']:.4f} s")
    print(f"  预估遮蔽时长 (简化模型): {params['occlusion_time']:.4f} s")
    return params

# --- 步骤 3: 最终精度验证 (与之前版本相同) ---
def step3_final_validation(final_params, n_points=1000):
    print(f"\n--- [步骤 3] 开始：使用{n_points}个目标点进行最终精度验证 ---")
    
    # 生成目标表面采样点
    points = []
    # (此处省略了生成点的代码，与上一版完全相同)
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), true_target_height * np.random.rand()
        points.append([true_target_base_center[0] + 7 * np.cos(theta), true_target_base_center[1] + 7 * np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = 7 * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), z])
    target_points = np.array(points)

    v, theta, t_drop, t_delay = final_params['v'], final_params['theta_rad'], final_params['t_drop'], final_params['t_delay']
    uav_velocity = np.array([np.cos(theta) * v, np.sin(theta) * v, 0])
    t_explode = t_drop + t_delay
    p_drop = uav_initial_pos + uav_velocity * t_drop
    v_drop = uav_velocity
    p_explode = p_drop + v_drop * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])

    total_occluded_time = 0
    time_step = 0.01
    for t in np.arange(t_explode, t_explode + SMOKE_DURATION, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        smoke_pos = p_explode + np.array([0, 0, -3.0 * (t - t_explode)])
        is_occluded_this_step = any(is_line_segment_intersecting_sphere(missile_pos, tp, smoke_pos, SMOKE_RADIUS) for tp in target_points)
        if is_occluded_this_step: total_occluded_time += time_step
            
    print("\n" + "="*50)
    print("  第二问最优策略及最终结果")
    print("="*50)
    print("  最优策略参数:")
    print(f"    无人机飞行速度: {v:.4f} m/s")
    print(f"    无人机飞行方向: {np.degrees(theta):.4f} 度")
    print(f"    投放点坐标: ({p_drop[0]:.2f}, {p_drop[1]:.2f}, {p_drop[2]:.2f})")
    print(f"    起爆点坐标: ({p_explode[0]:.2f}, {p_explode[1]:.2f}, {p_explode[2]:.2f})")
    print(f"  最终高精度有效遮蔽时长: {total_occluded_time:.4f} 秒")
    print("="*50)
    return total_occluded_time

# --- 主程序 ---
if __name__ == "__main__":
    start_total_time = time.time()
    
    initial_solution = step1_find_optimal_window()
    
    # 只有在第一步成功找到解时，才继续执行后续步骤
    if initial_solution:
        optimized_solution = step2_local_optimization(initial_solution)
        final_result = step3_final_validation(optimized_solution)
    
    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")
