import numpy as np
import time
import itertools
import pandas as pd
import multiprocessing
from tqdm import tqdm
from numba import njit
import warnings

warnings.filterwarnings('ignore')

GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
UAV_V_MIN, UAV_V_MAX = 70.0, 140.0
DROP_INTERVAL = 1.0  

P2_SEARCH_STEPS_COARSE = 1000
P3_OPTIMIZATION_ITERATIONS = 10
P4_OPTIMIZATION_ROUNDS = 10

UAV_NAMES = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
MISSILE_NAMES = ['M1', 'M2', 'M3']

UAV_INITIAL_POS = {
    'FY1': np.array([17800, 0, 1800], dtype=float),
    'FY2': np.array([12000, 1400, 1400], dtype=float),
    'FY3': np.array([6000, -3000, 700], dtype=float),
    'FY4': np.array([11000, 2000, 1800], dtype=float),
    'FY5': np.array([13000, -2000, 1300], dtype=float),
}

MISSILE_INITIAL_POS = {
    'M1': np.array([20000, 0, 2000], dtype=float),
    'M2': np.array([19000, 600, 2100], dtype=float),
    'M3': np.array([18000, -600, 1900], dtype=float),
}

false_target_pos = np.array([0, 0, 0], dtype=float)
true_target_base_center = np.array([0, 200, 0], dtype=float)
true_target_height = 10
simple_target_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
OCCLUSION_THRESHOLD_PERCENT = 70.0

MISSILE_INFO = {}
for name, pos in MISSILE_INITIAL_POS.items():
    velocity_vector = (false_target_pos - pos) / np.linalg.norm(false_target_pos - pos) * 300.0
    total_time = np.linalg.norm(false_target_pos - pos) / 300.0
    MISSILE_INFO[name] = {
        'initial_pos': pos,
        'velocity_vector': velocity_vector,
        'total_time': total_time
    }

@njit(fastmath=True)
def is_line_segment_intersecting_sphere_numba(p1, p2, sphere_center, sphere_radius):
    line_vec = p2 - p1
    point_vec = sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0:
        return np.sqrt(np.sum(point_vec**2)) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.maximum(0.0, np.minimum(1.0, t))
    closest_point_on_line = p1 + t * line_vec
    dist_sq = np.sum((sphere_center - closest_point_on_line)**2)
    return dist_sq <= sphere_radius**2

def get_occlusion_timeline(strategies, uav_pos_dict, missile_name, target_point, time_step=0.1):
    if not strategies: return 0
    missile_info = MISSILE_INFO[missile_name]
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
    
    total_occluded_time = 0
    time_points = np.arange(min_explode_time, max_end_time, time_step)

    for t_abs in time_points:
        missile_pos = missile_info['initial_pos'] + missile_info['velocity_vector'] * t_abs
        is_occluded_this_step = False
        for se in smoke_events:
            if se['t_explode'] <= t_abs < se['t_explode'] + SMOKE_DURATION:
                smoke_pos = se['p_explode'] + np.array([0, 0, -3.0 * (t_abs - se['t_explode'])])
                if is_line_segment_intersecting_sphere_numba(missile_pos, target_point, smoke_pos, SMOKE_RADIUS):
                    is_occluded_this_step = True
                    break
        if is_occluded_this_step:
            total_occluded_time += time_step
            
    return total_occluded_time

def solve_p2_style_worker(args):
    uav_name, missile_name = args
    uav_pos = UAV_INITIAL_POS[uav_name]
    missile_info = MISSILE_INFO[missile_name]
    
    feasible_solutions = []
    search_explode_times = np.linspace(0, missile_info['total_time'] * 0.9, P2_SEARCH_STEPS_COARSE)
    search_delays = np.linspace(0, 20, P2_SEARCH_STEPS_COARSE)
    search_los_ratios = np.linspace(0, 0.9, P2_SEARCH_STEPS_COARSE)

    for t_explode in search_explode_times:
        for t_delay in search_delays:
            t_drop = t_explode - t_delay
            if t_drop <= 0: continue
            
            missile_pos_at_explode = missile_info['initial_pos'] + missile_info['velocity_vector'] * t_explode
            
            for los_ratio in search_los_ratios:
                los_vector = simple_target_point - missile_pos_at_explode
                ideal_explode_pos = missile_pos_at_explode + los_ratio * los_vector
                
                required_vel_3d = (ideal_explode_pos - uav_pos - np.array([0, 0, -0.5 * GRAVITY * t_delay**2])) / t_explode
                required_vel_xy = required_vel_3d[:2]
                required_speed = np.linalg.norm(required_vel_xy)

                if UAV_V_MIN <= required_speed <= UAV_V_MAX:
                    strat = {
                        'uav_name': uav_name,
                        'v': required_speed,
                        'theta_rad': np.arctan2(required_vel_xy[1], required_vel_xy[0]),
                        't_drop': t_drop,
                        't_delay': t_delay
                    }
                    occlusion_time = get_occlusion_timeline([strat], {uav_name: uav_pos}, missile_name, simple_target_point)
                    if occlusion_time > 0.1:
                        feasible_solutions.append({**strat, 'occlusion_time': occlusion_time})

    if not feasible_solutions:
        return uav_name, missile_name, None, 0.0

    best_coarse_solution = max(feasible_solutions, key=lambda x: x['occlusion_time'])
    return uav_name, missile_name, best_coarse_solution, best_coarse_solution['occlusion_time']

def solve_p4_style(uav_names_in_squad, missile_name, initial_strategies):
    params = {s['uav_name']: s for s in initial_strategies}
    current_max_time = get_occlusion_timeline(list(params.values()), UAV_INITIAL_POS, missile_name, simple_target_point)

    for _ in range(P4_OPTIMIZATION_ROUNDS):
        last_occlusion_time = current_max_time
        for uav_name in uav_names_in_squad:
            for key in ['v', 'theta_rad', 't_drop', 't_delay']:
                original_val = params[uav_name][key]
                step = 2.0 if key == 'v' else np.radians(4) if key == 'theta_rad' else 0.4
                search_range = np.linspace(original_val - step, original_val + step, 5)
                
                for test_val in search_range:
                    if key == 'v' and not (UAV_V_MIN <= test_val <= UAV_V_MAX): continue
                    
                    test_params_list = []
                    for name in uav_names_in_squad:
                        if name == uav_name:
                            test_params_list.append({**params[name], key: test_val})
                        else:
                            test_params_list.append(params[name])
                    
                    t = get_occlusion_timeline(test_params_list, UAV_INITIAL_POS, missile_name, simple_target_point)
                    if t > current_max_time:
                        current_max_time = t
                        params[uav_name][key] = test_val
                        
        if current_max_time - last_occlusion_time < 0.01:
            break
            
    params['occlusion_time'] = current_max_time
    return params

def upgrade_to_three_smokes_worker(args):
    uav_name, single_smoke_params, missile_name = args
    
    t_drop_center = single_smoke_params['t_drop']
    base_delay = single_smoke_params['t_delay']
    t_drop_early = max(0.1, t_drop_center - DROP_INTERVAL - 0.2)
    t_drop_late = t_drop_center + DROP_INTERVAL + 0.2

    params = {
        'uav_name': uav_name,
        'v': single_smoke_params['v'],
        'theta_rad': single_smoke_params['theta_rad'],
        'drops': [
            {'t_drop': t_drop_early, 't_delay': base_delay},
            {'t_drop': t_drop_center, 't_delay': base_delay},
            {'t_drop': t_drop_late, 't_delay': base_delay},
        ]
    }
    
    def eval_3_smoke(p):
        strats = [{'uav_name': uav_name, 'v': p['v'], 'theta_rad': p['theta_rad'], **d} for d in p['drops']]
        return get_occlusion_timeline(strats, UAV_INITIAL_POS, missile_name, simple_target_point)

    current_max_time = eval_3_smoke(params)

    for _ in range(P3_OPTIMIZATION_ITERATIONS):
        last_occlusion_time = current_max_time
        for v_test in np.linspace(max(UAV_V_MIN, params['v'] - 5), min(UAV_V_MAX, params['v'] + 5), 5):
            test_params = params.copy(); test_params['v'] = v_test
            if (t := eval_3_smoke(test_params)) > current_max_time: current_max_time, params['v'] = t, v_test
        for theta_test in np.linspace(params['theta_rad'] - np.radians(8), params['theta_rad'] + np.radians(8), 5):
            test_params = params.copy(); test_params['theta_rad'] = theta_test
            if (t := eval_3_smoke(test_params)) > current_max_time: current_max_time, params['theta_rad'] = t, theta_test

        for i in range(3):
            original_drop = params['drops'][i]
            for t_drop_test in np.linspace(max(0.1, original_drop['t_drop'] - 1), original_drop['t_drop'] + 1, 5):
                test_params = params.copy(); test_params['drops'][i] = {'t_drop': t_drop_test, 't_delay': original_drop['t_delay']}
                if (t := eval_3_smoke(test_params)) > current_max_time: current_max_time, params['drops'][i]['t_drop'] = t, t_drop_test
            
            original_drop = params['drops'][i] # Re-fetch in case t_drop changed
            for t_delay_test in np.linspace(max(0.1, original_drop['t_delay'] - 1), original_drop['t_delay'] + 1, 5):
                test_params = params.copy(); test_params['drops'][i] = {'t_drop': original_drop['t_drop'], 't_delay': t_delay_test}
                if (t := eval_3_smoke(test_params)) > current_max_time: current_max_time, params['drops'][i]['t_delay'] = t, t_delay_test

        if current_max_time - last_occlusion_time < 0.01: break
            
    params['occlusion_time'] = current_max_time
    return uav_name, params

if __name__ == "__main__":
    start_total_time = time.time()
    
    print("="*80)
    print("        第五问：五机三弹协同干扰问题求解器启动")
    print("="*80)
    
    print("\n--- [阶段一] 开始：全局能力评估 (并行计算15种 UAV-Missile 组合) ---")
    tasks_p2 = [(uav, missile) for uav in UAV_NAMES for missile in MISSILE_NAMES]
    effectiveness_matrix = np.zeros((len(UAV_NAMES), len(MISSILE_NAMES)))
    all_single_smoke_solutions = {}

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(solve_p2_style_worker, tasks_p2), total=len(tasks_p2), desc="能力评估"))

    for uav_name, missile_name, sol, score in results:
        uav_idx = UAV_NAMES.index(uav_name)
        missile_idx = MISSILE_NAMES.index(missile_name)
        effectiveness_matrix[uav_idx, missile_idx] = score
        all_single_smoke_solutions[(uav_name, missile_name)] = sol

    print("\n能力评估矩阵 (行: UAV, 列: Missile):")
    df_eff = pd.DataFrame(effectiveness_matrix, index=UAV_NAMES, columns=MISSILE_NAMES)
    print(df_eff.round(4))

    squads = {m: [] for m in MISSILE_NAMES}
    assigned_uavs = set()

    all_potential_assignments = []
    for i, uav_name in enumerate(UAV_NAMES):
        for j, missile_name in enumerate(MISSILE_NAMES):
            score = effectiveness_matrix[i, j]
            if score > 0:
                all_potential_assignments.append({'score': score, 'uav': uav_name, 'missile': missile_name})
    
    all_potential_assignments.sort(key=lambda x: x['score'], reverse=True)
    
    covered_missiles = set()
    for assignment in all_potential_assignments:
        uav = assignment['uav']
        missile = assignment['missile']
        
        if uav not in assigned_uavs and missile not in covered_missiles:
            squads[missile].append(uav)
            assigned_uavs.add(uav)
            covered_missiles.add(missile)
            
        if len(covered_missiles) == len(MISSILE_NAMES):
            break
            
    remaining_uavs = [uav for uav in UAV_NAMES if uav not in assigned_uavs]
    
    for uav in remaining_uavs:
        best_target_for_secondary = None
        max_score = -1
        uav_idx = UAV_NAMES.index(uav)
        
        for j, missile in enumerate(MISSILE_NAMES):
            if len(squads[missile]) < 2:
                current_score = effectiveness_matrix[uav_idx, j]
                if current_score > max_score:
                    max_score = current_score
                    best_target_for_secondary = missile
        
        if best_target_for_secondary:
            squads[best_target_for_secondary].append(uav)
            assigned_uavs.add(uav)

    assignments_by_uav = {}
    for missile, uav_list in squads.items():
        for uav in uav_list:
            assignments_by_uav[uav] = missile
        
    print("\n--- [阶段一] 完成：贪心任务分配结果 ---")
    for missile, uavs in squads.items():
        print(f"  - 导弹 {missile} 由无人机 {uavs} 负责")
    
    end_total_time = time.time()
    print(f"\n程序总耗时: {end_total_time - start_total_time:.2f} 秒")