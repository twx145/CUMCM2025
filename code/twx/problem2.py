import numpy as np
import time
from tqdm import tqdm

# --- 1. Constants and Simulation Setup ---

# -- Precision & Search Grid Definition --
# 您可以调整这里的 _STEPS 来平衡速度和精度
# 更多的步骤 = 更精确但更慢
N_SPEED_STEPS = 15      # 速度 v 的采样点数
N_ANGLE_STEPS = 36      # 方向 θ 的采样点数
N_DROP_TIME_STEPS = 30  # 投放时间 t_drop 的采样点数
N_DELAY_TIME_STEPS = 15 # 延迟时间 t_delay 的采样点数

# -- Problem Constants --
NUM_TARGET_POINTS = 50  # 适当降低目标点数量以加快单次计算速度
TIME_STEP = 0.1         # 遮蔽计算的时间步长
GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
MISSILE_SPEED = 300.0

# -- Initial Positions & Target Geometry --
uav_initial_pos = np.array([17800, 0, 1800])
missile_initial_pos = np.array([20000, 0, 2000])
false_target_pos = np.array([0, 0, 0])
true_target_base_center = np.array([0, 200, 0])
true_target_radius = 7
true_target_height = 10

# --- 2. Helper Functions (Slightly Optimized) ---

# Pre-generate target points once to save time
TARGET_POINTS = None
def get_target_points():
    global TARGET_POINTS
    if TARGET_POINTS is None:
        points = []
        n_side, n_caps = int(NUM_TARGET_POINTS * 0.6), int(NUM_TARGET_POINTS * 0.2)
        for i in range(n_side):
            theta, z = 2 * np.pi * (i / n_side), true_target_height * np.random.rand()
            points.append([true_target_base_center[0] + true_target_radius * np.cos(theta), true_target_base_center[1] + true_target_radius * np.sin(theta), z])
        for i in range(n_caps * 2):
            r, theta = true_target_radius * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
            z = 0 if i < n_caps else true_target_height
            points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), true_target_base_center[2] + z])
        TARGET_POINTS = np.array(points)
    return TARGET_POINTS

def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius_sq):
    if np.any(np.isnan(sphere_center)): return False
    line_vec, point_vec = p2 - p1, sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.dot(point_vec, point_vec) <= sphere_radius_sq
    t = np.dot(point_vec, line_vec) / line_len_sq
    closest_point = p1 + np.clip(t, 0, 1) * line_vec
    return np.sum((sphere_center - closest_point)**2) <= sphere_radius_sq

# --- 3. Core Occlusion Calculator (The "Objective Function") ---

def calculate_occlusion_for_strategy(uav_velocity, drop_time, delay_time):
    """A streamlined calculator that takes a strategy and returns occlusion time."""
    target_points = get_target_points()
    missile_velocity = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * MISSILE_SPEED
    
    explode_time = drop_time + delay_time
    drop_pos = uav_initial_pos + uav_velocity * drop_time
    explode_pos = drop_pos + uav_velocity * delay_time + np.array([0, 0, -0.5 * GRAVITY * delay_time**2])

    total_occluded_time = 0.0
    simulation_start_time = explode_time
    simulation_end_time = explode_time + SMOKE_DURATION
    smoke_radius_sq = SMOKE_RADIUS**2

    for t in np.arange(simulation_start_time, simulation_end_time, TIME_STEP):
        missile_pos = missile_initial_pos + missile_velocity * t
        smoke_center_pos = explode_pos + np.array([0, 0, -3.0 * (t - explode_time)])
        
        for point in target_points:
            if is_line_segment_intersecting_sphere(missile_pos, point, smoke_center_pos, smoke_radius_sq):
                total_occluded_time += TIME_STEP
                break
    return total_occluded_time

# --- 4. The Grid Search Optimizer ---

def find_optimal_strategy():
    """Performs a grid search to find the best strategy."""
    
    # Define the search space
    speeds = np.linspace(70, 140, N_SPEED_STEPS)
    angles = np.linspace(0, 2 * np.pi, N_ANGLE_STEPS)
    drop_times = np.linspace(1.0, 60.0, N_DROP_TIME_STEPS)
    delay_times = np.linspace(2.0, 15.0, N_DELAY_TIME_STEPS)
    
    best_occlusion_time = -1.0
    best_params = {}
    
    total_iterations = len(speeds) * len(angles) * len(drop_times) * len(delay_times)
    print(f"Starting grid search with {total_iterations} total combinations.")
    
    progress_bar = tqdm(total=total_iterations, unit="eval", desc="Optimizing Strategy")
    
    for v in speeds:
        for theta in angles:
            # Create velocity vector from speed and angle (on xy-plane)
            uav_vel = np.array([v * np.cos(theta), v * np.sin(theta), 0])
            for t_drop in drop_times:
                for t_delay in delay_times:
                    occlusion_time = calculate_occlusion_for_strategy(uav_vel, t_drop, t_delay)
                    
                    if occlusion_time > best_occlusion_time:
                        best_occlusion_time = occlusion_time
                        best_params = {
                            'speed': v,
                            'angle_deg': np.rad2deg(theta),
                            'drop_time': t_drop,
                            'delay_time': t_delay,
                        }
                    progress_bar.update(1)

    progress_bar.close()
    
    # Calculate derived results for the final report
    if best_params:
        v_best = best_params['speed']
        theta_best = np.deg2rad(best_params['angle_deg'])
        t_drop_best = best_params['drop_time']
        t_delay_best = best_params['delay_time']

        best_vel_vec = np.array([v_best * np.cos(theta_best), v_best * np.sin(theta_best), 0])
        drop_point = uav_initial_pos + best_vel_vec * t_drop_best
        explode_point = drop_point + best_vel_vec * t_delay_best + np.array([0, 0, -0.5 * GRAVITY * t_delay_best**2])
        best_params['drop_point'] = drop_point
        best_params['explode_point'] = explode_point

    return best_occlusion_time, best_params

# --- 5. Run the optimizer ---

if __name__ == "__main__":
    start_time = time.time()
    
    # Pre-generate points before starting the main loop
    get_target_points()
    
    optimal_time, optimal_params = find_optimal_strategy()
    
    end_time = time.time()
    
    print("\n" + "="*50)
    print("  OPTIMIZATION COMPLETE")
    print(f"  Total Runtime: {end_time - start_time:.2f} seconds")
    print("="*50)
    if optimal_params:
        print(f"  Maximal Occlusion Time Found: {optimal_time:.4f} seconds")
        print("\n  Optimal Strategy Parameters:")
        print(f"  - UAV Speed (v):               {optimal_params['speed']:.2f} m/s")
        print(f"  - UAV Direction (θ):           {optimal_params['angle_deg']:.2f} degrees")
        print(f"  - Grenade Drop Time (t_drop):  {optimal_params['drop_time']:.2f} s")
        print(f"  - Explosion Delay (t_delay):   {optimal_params['delay_time']:.2f} s")
        print("\n  Derived Optimal Locations:")
        print(f"  - Drop Point (x,y,z):      ({optimal_params['drop_point'][0]:.1f}, {optimal_params['drop_point'][1]:.1f}, {optimal_params['drop_point'][2]:.1f})")
        print(f"  - Explosion Point (x,y,z): ({optimal_params['explode_point'][0]:.1f}, {optimal_params['explode_point'][1]:.1f}, {optimal_params['explode_point'][2]:.1f})")
        print("="*50)
    else:
        print("  No solution found.")