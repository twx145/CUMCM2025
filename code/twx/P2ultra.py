import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0

NUM_INITIAL_STARTS = 500

OPTIMIZATION_ITERATIONS = 10

OCCLUSION_THRESHOLD_PERCENT = 70.0

uav_initial_pos = np.array([17800, 0, 1800], dtype=float)
missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])

missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * 300.0
missile_total_time = np.linalg.norm(false_target_pos - missile_initial_pos) / 300.0

SEARCH_EXPLODE_TIMES_RANGE = (missile_total_time * 0, missile_total_time * 0.9)
SEARCH_EXPLODE_TIMES_STEPS = 200
SEARCH_DELAYS_RANGE = (0, 8.0)
SEARCH_DELAYS_STEPS = 150
SEARCH_LOS_RATIO_RANGE = (0, 0.9)
SEARCH_LOS_RATIO_STEPS = 90

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
    time_step = 0.01
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

def step1_find_top_N_windows(n):
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
                if current_occlusion_time > 0:
                    feasible_solutions.append({'v': req_speed, 'theta_rad': req_dir_rad, 't_drop': t_drop, 't_delay': t_delay, 'occlusion_time': current_occlusion_time})
    if not feasible_solutions: return []
    sorted_solutions = sorted(feasible_solutions, key=lambda x: x['occlusion_time'], reverse=True)
    top_n_solutions = []
    for sol in sorted_solutions:
        if not any(np.linalg.norm([sol['v'] - ts['v'], (sol['t_drop']-ts['t_drop'])*10]) < 5.0 for ts in top_n_solutions):
            top_n_solutions.append(sol)
        if len(top_n_solutions) >= n: break
    print(f"步骤1完成。从{len(feasible_solutions)}个可行解中筛选出 {len(top_n_solutions)} 个高质量初始解。")
    return top_n_solutions

def step2_local_optimization(initial_params):
    params = initial_params.copy()
    convergence_history = [params['occlusion_time']] 

    for i in range(OPTIMIZATION_ITERATIONS):
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
        
        convergence_history.append(params['occlusion_time'])
        if params['occlusion_time'] - last_occlusion_time < 0.001: break
    return params, convergence_history

def step3_final_validation(final_params, n_points=1000, threshold_percent=70.0):
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points)); points = []
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), true_target_height * np.random.rand()
        points.append([true_target_base_center[0] + 7 * np.cos(theta), true_target_base_center[1] + 7 * np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = 7 * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), z])
    target_points = np.array(points)
    v, theta_rad, t_drop, t_delay = final_params['v'], final_params['theta_rad'], final_params['t_drop'], final_params['t_delay']
    uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
    t_explode = t_drop + t_delay
    p_drop = uav_initial_pos + uav_velocity * t_drop
    p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
    total_occluded_time = 0.0
    time_step = 0.001
    for t in np.arange(t_explode, t_explode + SMOKE_DURATION, time_step):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        smoke_pos = p_explode + np.array([0, 0, -3.0 * (t - t_explode)])
        occluded_lines_count = sum(1 for tp in target_points if is_line_segment_intersecting_sphere(missile_pos, tp, smoke_pos, SMOKE_RADIUS))
        if occluded_lines_count >= required_occluded_count:
            total_occluded_time += time_step
    return total_occluded_time, final_params, p_drop, p_explode

def save_results_to_csv(results_list, initial_list, histories_list, filename="problem2_optimization_results.csv"):
    base_header = ['start_point_id', 'initial_score', 'optimized_score', 'v', 'theta_deg', 't_drop', 't_delay']
    iter_header = [f'iter_{i}_score' for i in range(OPTIMIZATION_ITERATIONS + 1)]
    header = base_header + iter_header
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (res, initial, history) in enumerate(zip(results_list, initial_list, histories_list)):
            padded_history = history + [history[-1]] * (OPTIMIZATION_ITERATIONS + 1 - len(history))
            row = [
                i + 1,
                f"{initial['occlusion_time']:.4f}",
                f"{res['occlusion_time']:.4f}",
                f"{res['v']:.4f}",
                f"{np.degrees(res['theta_rad']):.4f}",
                f"{res['t_drop']:.4f}",
                f"{res['t_delay']:.4f}",
            ]
            row.extend([f"{h:.4f}" for h in padded_history])
            writer.writerow(row)
    print(f"\n所有 {len(results_list)} 个优化结果及其收敛历史已保存到文件: {filename}")

def visualize_optimization_journey(initial_list, final_list):
    print("\n--- 正在生成4维解空间降维可视化图 ---")
    def params_to_vector(params):
        return [params['v'], np.degrees(params['theta_rad']), params['t_drop'], params['t_delay']]
    initial_vectors = np.array([params_to_vector(p) for p in initial_list])
    final_vectors = np.array([params_to_vector(p) for p in final_list])
    scaler = StandardScaler(); all_vectors_scaled = scaler.fit_transform(np.vstack([initial_vectors, final_vectors]))
    pca = PCA(n_components=2); all_vectors_pca = pca.fit_transform(all_vectors_scaled)
    initial_pca, final_pca = all_vectors_pca[:len(initial_list)], all_vectors_pca[len(initial_list):]
    best_idx = np.argmax([p['occlusion_time'] for p in final_list])
    plt.figure(figsize=(12, 9))
    for i in range(len(initial_list)):
        plt.arrow(initial_pca[i, 0], initial_pca[i, 1], final_pca[i, 0] - initial_pca[i, 0], final_pca[i, 1] - initial_pca[i, 1], head_width=0.1, fc='gray', ec='gray', alpha=0.6, length_includes_head=True)
    plt.scatter(initial_pca[:, 0], initial_pca[:, 1], c='blue', s=100, alpha=0.8, label='初始解 (Starts)')
    plt.scatter(final_pca[:, 0], final_pca[:, 1], c='red', s=100, alpha=0.8, label='局部最优解 (Local Optima)')
    plt.scatter(final_pca[best_idx, 0], final_pca[best_idx, 1], c='green', s=300, marker='*', edgecolors='black', label='最佳候选解 (Best Candidate)')
    plt.title('4维解空间的PCA降维可视化 (问题二)', fontsize=16); plt.xlabel('主成分 1'); plt.ylabel('主成分 2')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.show()


if __name__ == "__main__":
    start_total_time = time.time()
    
    initial_solutions_list = step1_find_top_N_windows(n=NUM_INITIAL_STARTS)
    
    if not initial_solutions_list:
        print("错误：第一步未能找到任何可行的初始解。")
    else:
        local_optima_list, histories_list = [], []
        for i, initial_sol in enumerate(initial_solutions_list):
            print(f"\n--- 对第 {i+1}/{len(initial_solutions_list)} 个初始解进行局部优化 (初始时长: {initial_sol['occlusion_time']:.2f}s) ---")
            optimized_solution, history = step2_local_optimization(initial_sol)
            local_optima_list.append(optimized_solution)
            histories_list.append(history)
            print(f"--- 优化完成，得到的局部最优时长: {optimized_solution['occlusion_time']:.4f}s ---")
        
        save_results_to_csv(local_optima_list, initial_solutions_list, histories_list)
        visualize_optimization_journey(initial_solutions_list, local_optima_list)

        best_overall_solution = max(local_optima_list, key=lambda x: x['occlusion_time'])
        
        print("\n--- 对全局最优候选解进行最终高精度验证 ---")
        final_time, final_params, p_drop, p_explode = step3_final_validation(
            best_overall_solution, 
            threshold_percent=OCCLUSION_THRESHOLD_PERCENT
        )
        
        
        v, theta_rad, t_drop, t_delay = final_params['v'], final_params['theta_rad'], final_params['t_drop'], final_params['t_delay']
        t_explode = t_drop + t_delay
        print("\n" + "="*60)
        print("        第二问：最优干扰策略及最终结果 (多起点优化)")
        print("="*60)
        print(f"\n搜索概况: 从 {len(initial_solutions_list)} 个高质量起点出发，找到的最佳策略如下。")
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
        print(f"  遮蔽有效性阈值          : {OCCLUSION_THRESHOLD_PERCENT:.1f}%")
        print(f"  最终高精度有效遮蔽时长  : {final_time:.4f} 秒")
        print("="*60)

    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")