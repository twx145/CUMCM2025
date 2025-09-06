import numpy as np
import time
import itertools
from collections import defaultdict
import warnings
import multiprocessing
from tqdm import tqdm
from numba import njit

# 忽略由于搜索空间离散化可能产生的数值计算警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. 可配置参数 (Parameters You Can Control)
# ==============================================================================
# [阶段一] 策略库大小：为每个UAV-Missile组合生成的顶尖单弹策略数量
TOP_N_STRATEGIES = 50

# [阶段二] 任务分配数量：贪心算法尝试分配的最大弹药总数 (上限为 5 UAV * 3 = 15)
NUM_GRENADES_TO_ALLOCATE = 15

# [阶段三] 局部优化轮数：对贪心规划结果进行协同微调的迭代次数
NUM_OPTIMIZATION_ROUNDS = 10

# ==============================================================================
# 2. 问题常量与环境设置 (Problem Constants & Environment Setup)
# ==============================================================================
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0
MISSILE_SPEED = 300.0
DROP_INTERVAL = 1.0
FINAL_VALIDATION_THRESHOLD_PERCENT = 70.0

# 实体名称与初始位置
UAV_DATA = {
    'FY1': {'pos': np.array([17800, 0, 1800], dtype=float)},
    'FY2': {'pos': np.array([12000, 1400, 1400], dtype=float)},
    'FY3': {'pos': np.array([6000, -3000, 700], dtype=float)},
    'FY4': {'pos': np.array([11000, 2000, 1800], dtype=float)},
    'FY5': {'pos': np.array([13000, -2000, 1300], dtype=float)},
}
MISSILE_DATA = {
    'M1': {'pos': np.array([20000, 0, 2000], dtype=float)},
    'M2': {'pos': np.array([19000, 600, 2100], dtype=float)},
    'M3': {'pos': np.array([18000, -600, 1900], dtype=float)},
}

FALSE_TARGET_POS = np.array([0, 0, 0], dtype=float)
TRUE_TARGET_BASE_CENTER = np.array([0, 200, 0], dtype=float)
TRUE_TARGET_HEIGHT = 10.0
SIMPLE_TARGET_POINT = TRUE_TARGET_BASE_CENTER + np.array([0, 0, TRUE_TARGET_HEIGHT / 2])

for m_name, m_info in MISSILE_DATA.items():
    vec_to_target = FALSE_TARGET_POS - m_info['pos']
    dist_to_target = np.linalg.norm(vec_to_target)
    m_info['velocity_vec'] = vec_to_target / dist_to_target * MISSILE_SPEED
    m_info['total_time'] = dist_to_target / MISSILE_SPEED

# ==============================================================================
# 3. 核心计算与辅助函数 (Core Calculation & Helper Functions)
# ==============================================================================

# --- Numba 加速的核心数学函数 ---
@njit(fastmath=True)
def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius):
    """[Numba JIT加速] 检查线段p1-p2是否与球体相交"""
    line_vec = p2 - p1
    point_vec = sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq == 0.0:
        return np.sqrt(np.sum(point_vec**2)) <= sphere_radius
        
    t = np.dot(point_vec, line_vec) / line_len_sq
    
    # 手动实现 clip(t, 0, 1)
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0
        
    closest_point_on_line = p1 + t * line_vec
    dist_sq = np.sum((sphere_center - closest_point_on_line)**2)
    
    return dist_sq <= sphere_radius**2

def get_total_occlusion_time(plan, missile_name, time_step=0.1):
    """计算给定规划对特定导弹的总遮蔽时长（处理时间并集）"""
    # ... (此函数逻辑不变，但其内部调用的 is_line_segment_intersecting_sphere 已被Numba加速)
    if not plan: return 0.0
    missile_info = MISSILE_DATA[missile_name]
    smoke_events = []
    min_explode_time, max_end_time = float('inf'), float('-inf')
    for strat in plan:
        if strat['missile_target'] != missile_name: continue
        uav_pos = UAV_DATA[strat['uav_name']]['pos']
        v, theta_rad, t_drop, t_delay = strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']
        uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
        t_explode = t_drop + t_delay
        p_drop = uav_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        smoke_events.append({'t_explode': t_explode, 'p_explode': p_explode})
        min_explode_time = min(min_explode_time, t_explode)
        max_end_time = max(max_end_time, t_explode + SMOKE_DURATION)

    if not smoke_events: return 0.0
    timeline_start = min_explode_time
    timeline_end = max_end_time
    timeline_len = int((timeline_end - timeline_start) / time_step) + 1
    occlusion_timeline = np.zeros(timeline_len, dtype=bool)
    for t_idx, t_abs in enumerate(np.arange(timeline_start, timeline_end, time_step)):
        if t_abs > missile_info['total_time']: break
        missile_pos = missile_info['pos'] + missile_info['velocity_vec'] * t_abs
        for se in smoke_events:
            if se['t_explode'] <= t_abs < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -SMOKE_SINK_SPEED * (t_abs - se['t_explode'])])
                if is_line_segment_intersecting_sphere(missile_pos, SIMPLE_TARGET_POINT, smoke_pos, SMOKE_RADIUS):
                    occlusion_timeline[t_idx] = True
                    break
    return np.sum(occlusion_timeline) * time_step

# ==============================================================================
# 4. 算法框架各阶段实现 (Implementation of Algorithm Stages)
# ==============================================================================

# --- [并行化改造] 阶段一的工作函数，用于单个CPU核心 ---
def calculate_single_pair_strategies(args):
    """[并行Worker] 计算单个UAV-Missile组合的策略库"""
    uav_name, m_name, top_n = args
    uav_info = UAV_DATA[uav_name]
    m_info = MISSILE_DATA[m_name]
    
    feasible_solutions = []
    search_explode_times = np.linspace(0, m_info['total_time'] * 0.8, 200)
    search_delays = np.linspace(0, 8.0, 400)
    search_los_ratios = np.linspace(0, 0.9, 200)

    for t_explode in search_explode_times:
        for t_delay in search_delays:
            t_drop = t_explode - t_delay
            if t_drop <= 0: continue
            missile_pos_at_explode = m_info['pos'] + m_info['velocity_vec'] * t_explode
            for los_ratio in search_los_ratios:
                los_vector = SIMPLE_TARGET_POINT - missile_pos_at_explode
                ideal_explode_pos = missile_pos_at_explode + los_ratio * los_vector
                required_vel_3d = (ideal_explode_pos - uav_info['pos'] - np.array([0, 0, -0.5 * GRAVITY * t_delay**2])) / t_explode
                required_vel_xy = required_vel_3d[:2]
                required_speed = np.linalg.norm(required_vel_xy)
                if UAV_V_MIN <= required_speed <= UAV_V_MAX:
                    strat = {'uav_name': uav_name, 'missile_target': m_name, 'v': required_speed, 
                             'theta_rad': np.arctan2(required_vel_xy[1], required_vel_xy[0]),
                             't_drop': t_drop, 't_delay': t_delay}
                    strat['occlusion_time'] = get_total_occlusion_time([strat], m_name)
                    if strat['occlusion_time'] > 1.0:
                        feasible_solutions.append(strat)
    
    sorted_solutions = sorted(feasible_solutions, key=lambda x: x['occlusion_time'], reverse=True)
    return uav_name, m_name, sorted_solutions[:top_n]

# --- [并行化改造] 阶段一的协调函数，负责分发任务 ---
def stage1_build_strategy_library_parallel(top_n):
    """[并行Coordinator] 创建并分发任务给所有CPU核心来构建策略库"""
    print("\n--- [阶段一] 开始：构建基础策略效能库 (CPU多核并行) ---")
    
    tasks = [(u_name, m_name, top_n) for u_name in UAV_DATA.keys() for m_name in MISSILE_DATA.keys()]

    library = defaultdict(lambda: defaultdict(list))
    # 使用所有可用的CPU核心
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # 使用tqdm显示进度条
        results = list(tqdm(pool.imap_unordered(calculate_single_pair_strategies, tasks), total=len(tasks), desc="构建策略库"))
    
    for uav_name, m_name, strategies in results:
        library[uav_name][m_name] = strategies
        
    print("--- [阶段一] 完成：策略库构建完毕 ---")
    return library

# --- 阶段二、三、四的函数保持不变 ---
def stage2_greedy_task_allocation(library, max_allocations):
    # ... (此函数代码与上一版完全相同)
    print("\n--- [阶段二] 开始：基于边际增益的贪心任务分配 ---")
    missile_occlusion_times = {m_name: 0.0 for m_name in MISSILE_DATA}
    uav_grenade_count = defaultdict(int)
    uav_locked_params = {}
    final_plan = []
    for i in range(max_allocations):
        worst_missile = min(missile_occlusion_times, key=missile_occlusion_times.get)
        best_gain = -1.0
        best_strategy_to_add = None
        for uav_name in UAV_DATA.keys():
            for candidate_strat in library[uav_name][worst_missile]:
                if uav_grenade_count[uav_name] >= 3: continue
                if uav_name in uav_locked_params:
                    locked = uav_locked_params[uav_name]
                    if not (np.isclose(candidate_strat['v'], locked['v']) and 
                            np.isclose(candidate_strat['theta_rad'], locked['theta_rad'])):
                        continue
                last_drop_time = -1.0
                for planned_strat in final_plan:
                    if planned_strat['uav_name'] == uav_name:
                        last_drop_time = max(last_drop_time, planned_strat['t_drop'])
                if candidate_strat['t_drop'] < last_drop_time + DROP_INTERVAL: continue
                potential_plan = final_plan + [candidate_strat]
                new_time = get_total_occlusion_time(potential_plan, worst_missile)
                gain = new_time - missile_occlusion_times[worst_missile]
                if gain > best_gain:
                    best_gain = gain
                    best_strategy_to_add = candidate_strat
        if best_strategy_to_add:
            print(f"  - 分配第 {i+1} 枚弹: {best_strategy_to_add['uav_name']} -> {worst_missile} (增益: {best_gain:.2f}s)")
            final_plan.append(best_strategy_to_add)
            uav_name = best_strategy_to_add['uav_name']
            uav_grenade_count[uav_name] += 1
            if uav_name not in uav_locked_params:
                uav_locked_params[uav_name] = {'v': best_strategy_to_add['v'], 'theta_rad': best_strategy_to_add['theta_rad']}
            for m_name in MISSILE_DATA:
                missile_occlusion_times[m_name] = get_total_occlusion_time(final_plan, m_name)
        else:
            print(f"  - 在第 {i+1} 次分配时未找到有效增益策略，分配结束。")
            break
    print("--- [阶段二] 完成：贪心任务规划完毕 ---")
    return final_plan

def stage3_coordinated_local_optimization(plan, rounds):
    # ... (此函数代码与上一版完全相同)
    print("\n--- [阶段三] 开始：高维协同局部精调 ---")
    if not plan:
        print("  - 计划为空，跳过优化。")
        return []
    optimized_plan = [s.copy() for s in plan]
    for r in range(rounds):
        current_min_occlusion = min(get_total_occlusion_time(optimized_plan, m) for m in MISSILE_DATA)
        improved = False
        for i, strat_to_tweak in enumerate(optimized_plan):
            uav_name = strat_to_tweak['uav_name']
            for param_key, step, bounds in [('v', 5.0, (UAV_V_MIN, UAV_V_MAX)), ('theta_rad', np.radians(10), (-np.pi, np.pi))]:
                original_val = strat_to_tweak[param_key]
                for test_val in np.linspace(original_val - step, original_val + step, 5):
                    if not (bounds[0] <= test_val <= bounds[1]): continue
                    temp_plan = [s.copy() for s in optimized_plan]
                    for s in temp_plan:
                        if s['uav_name'] == uav_name: s[param_key] = test_val
                    new_min_occlusion = min(get_total_occlusion_time(temp_plan, m) for m in MISSILE_DATA)
                    if new_min_occlusion > current_min_occlusion:
                        optimized_plan = temp_plan
                        current_min_occlusion = new_min_occlusion
                        improved = True
            for param_key, step in [('t_drop', 1.0), ('t_delay', 1.0)]:
                original_val = strat_to_tweak[param_key]
                for test_val in np.linspace(max(0.1, original_val - step), original_val + step, 5):
                    temp_plan = [s.copy() for s in optimized_plan]
                    temp_plan[i][param_key] = test_val
                    uav_drops = sorted([s['t_drop'] for s in temp_plan if s['uav_name'] == uav_name])
                    if any(uav_drops[j+1] - uav_drops[j] < DROP_INTERVAL for j in range(len(uav_drops)-1)): continue
                    new_min_occlusion = min(get_total_occlusion_time(temp_plan, m) for m in MISSILE_DATA)
                    if new_min_occlusion > current_min_occlusion:
                        optimized_plan = temp_plan
                        current_min_occlusion = new_min_occlusion
                        improved = True
        print(f"  - 第 {r+1}/{rounds} 轮优化完成。当前瓶颈遮蔽时长: {current_min_occlusion:.4f}s")
        if not improved:
            print("  - 未能进一步提升性能，优化提前结束。")
            break
    print("--- [阶段三] 完成：局部优化完毕 ---")
    return optimized_plan

def stage4_final_validation(final_plan):
    # ... (此函数代码与上一版完全相同)
    print("\n--- [阶段四] 开始：最终高精度验证 ---")
    if not final_plan:
        print("  - 计划为空，无法验证。")
        return {}, {}
    n_points = 1000
    required_occluded_count = int(np.ceil((FINAL_VALIDATION_THRESHOLD_PERCENT / 100.0) * n_points))
    target_points = []
    for _ in range(int(n_points * 0.7)):
        angle, z = 2 * np.pi * np.random.rand(), TRUE_TARGET_HEIGHT * np.random.rand()
        target_points.append([TRUE_TARGET_BASE_CENTER[0] + 7 * np.cos(angle), TRUE_TARGET_BASE_CENTER[1] + 7 * np.sin(angle), z])
    for _ in range(int(n_points * 0.3)):
        r, angle = 7 * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else TRUE_TARGET_HEIGHT
        target_points.append([TRUE_TARGET_BASE_CENTER[0] + r * np.cos(angle), TRUE_TARGET_BASE_CENTER[1] + r * np.sin(angle), z])
    target_points = np.array(target_points)
    smoke_events, detailed_results = [], defaultdict(list)
    min_event_time, max_event_time = float('inf'), float('-inf')
    for strat in final_plan:
        uav_pos = UAV_DATA[strat['uav_name']]['pos']
        v, theta_rad, t_drop, t_delay = strat['v'], strat['theta_rad'], strat['t_drop'], strat['t_delay']
        uav_velocity = np.array([np.cos(theta_rad) * v, np.sin(theta_rad) * v, 0])
        t_explode = t_drop + t_delay
        p_drop = uav_pos + uav_velocity * t_drop
        p_explode = p_drop + uav_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        event = {'t_explode': t_explode, 'p_explode': p_explode, 'missile_target': strat['missile_target']}
        smoke_events.append(event)
        detailed_results[strat['uav_name']].append({**strat, 'p_drop': p_drop, 'p_explode': p_explode})
        min_event_time = min(min_event_time, t_explode)
        max_event_time = max(max_event_time, t_explode + SMOKE_DURATION)
    final_occlusion_times = {m_name: 0.0 for m_name in MISSILE_DATA}
    time_step = 0.01
    for t in np.arange(min_event_time, max_event_time, time_step):
        active_smokes_by_missile = defaultdict(list)
        for se in smoke_events:
            if se['t_explode'] <= t < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -SMOKE_SINK_SPEED * (t - se['t_explode'])])
                active_smokes_by_missile[se['missile_target']].append(smoke_pos)
        for m_name, m_info in MISSILE_DATA.items():
            if t > m_info['total_time'] or not active_smokes_by_missile[m_name]: continue
            missile_pos = m_info['pos'] + m_info['velocity_vec'] * t
            occluded_lines_count = 0
            for tp in target_points:
                for smoke_pos in active_smokes_by_missile[m_name]:
                    if is_line_segment_intersecting_sphere(missile_pos, tp, smoke_pos, SMOKE_RADIUS):
                        occluded_lines_count += 1
                        break
            if occluded_lines_count >= required_occluded_count:
                final_occlusion_times[m_name] += time_step
    print("--- [阶段四] 完成：高精度验证完毕 ---")
    return final_occlusion_times, detailed_results

# ==============================================================================
# 5. 主程序入口 (Main Execution Block)
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # --- 执行四阶段算法 (调用并行化版本) ---
    strategy_library = stage1_build_strategy_library_parallel(top_n=TOP_N_STRATEGIES)
    initial_plan = stage2_greedy_task_allocation(library=strategy_library, max_allocations=NUM_GRENADES_TO_ALLOCATE)
    optimized_plan = stage3_coordinated_local_optimization(plan=initial_plan, rounds=NUM_OPTIMIZATION_ROUNDS)
    final_times, final_details = stage4_final_validation(final_plan=optimized_plan)

    # --- 格式化并打印最终结果 ---
    print("\n" + "="*80)
    print(" " * 25 + "第五问：最终最优协同干扰策略")
    print("="*80)
    if not final_details:
        print("\n未能生成有效的干扰计划。")
    else:
        print("\n--- 最终性能评估 (高精度) ---")
        for m_name, t in final_times.items():
            print(f"  - 导弹 {m_name} 有效遮蔽时长: {t:.4f} 秒")
        bottleneck_time = min(final_times.values()) if final_times else 0.0
        print(f"\n  >> 目标函数值 (瓶颈时长): {bottleneck_time:.4f} 秒 <<")
        print("\n" + "-"*80)
        print("\n--- 各无人机详细执行方案 ---")
        sorted_uavs = sorted(final_details.keys())
        for uav_name in sorted_uavs:
            details = final_details[uav_name]
            sorted_tasks = sorted(details, key=lambda x: x['t_drop'])
            print(f"\n[无人机: {uav_name}]")
            print(f"  - 统一飞行速度 (v)      : {sorted_tasks[0]['v']:.4f} m/s")
            print(f"  - 统一飞行方向 (θ)      : {np.degrees(sorted_tasks[0]['theta_rad']):.4f} 度")
            for i, task in enumerate(sorted_tasks):
                print(f"  - 第 {i+1} 枚弹 (干扰目标: {task['missile_target']}):")
                print(f"    - 投放时间 (t_drop)   : {task['t_drop']:.4f} 秒")
                print(f"    - 引信延迟 (t_delay)    : {task['t_delay']:.4f} 秒")
                print(f"    - 起爆时间 (t_explode)  : {task['t_drop'] + task['t_delay']:.4f} 秒")
                p_drop, p_exp = task['p_drop'], task['p_explode']
                print(f"    - 投放点坐标 (P_drop)   : ({p_drop[0]:.2f}, {p_drop[1]:.2f}, {p_drop[2]:.2f})")
                print(f"    - 起爆点坐标 (P_explode): ({p_exp[0]:.2f}, {p_exp[1]:.2f}, {p_exp[2]:.2f})")
    
    print("\n" + "="*80)
    end_time = time.time()
    print(f"\n程序总耗时: {end_time - start_time:.2f} 秒")