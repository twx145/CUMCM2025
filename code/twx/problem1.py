import numpy as np
import time

# --- 1. Simulation and Precision Parameters ---

# -- Core Settings --
NUM_TARGET_POINTS = 1000       # Number of sample points on the target surface.
TIME_STEP = 0.01              # Simulation time step in seconds for accuracy.

# -- !!! NEW: ADJUSTABLE THRESHOLD !!! --
# Set the percentage of sight lines that must be blocked to be considered "effective occlusion".
# 0.1 means at least 1 line must be blocked (similar to the previous script).
# 50.0 means 50% or more of the lines must be blocked.
# 100.0 means all N lines must be blocked.
OCCLUSION_THRESHOLD_PERCENT = 50.0

# -- Physical and Strategic Constants (Problem 1) --
GRAVITY = 9.8
UAV_SPEED = 120.0
DROP_TIME = 1.5
EXPLODE_DELAY = 3.6
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
MISSILE_SPEED = 300.0

# -- Initial Positions --
uav_initial_pos = np.array([17800, 0, 1800])
missile_initial_pos = np.array([20000, 0, 2000])
false_target_pos = np.array([0, 0, 0])

# -- True Target Geometry --
true_target_base_center = np.array([0, 200, 0])
true_target_radius = 7
true_target_height = 10


# --- 2. Helper Functions (Unchanged from previous version) ---

def generate_target_points(n_points, base_center, radius, height):
    """Generates a set of points on the surface of a cylinder."""
    points = []
    n_side = int(n_points * 0.6)
    n_caps = int(n_points * 0.2)
    for i in range(n_side):
        theta, z = 2 * np.pi * (i / n_side), height * np.random.rand()
        points.append([base_center[0] + radius * np.cos(theta), base_center[1] + radius * np.sin(theta), z])
    for i in range(n_caps * 2): # Combine loop for top and bottom
        r, theta = radius * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if i < n_caps else height # First half on bottom, second half on top
        points.append([base_center[0] + r * np.cos(theta), base_center[1] + r * np.sin(theta), base_center[2] + z])
    return np.array(points)

def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius):
    """Checks if the line segment from p1 to p2 intersects with a sphere."""
    if np.any(np.isnan(sphere_center)): return False
    line_vec, point_vec = p2 - p1, sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.linalg.norm(point_vec) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    closest_point = p1 + np.clip(t, 0, 1) * line_vec
    return np.linalg.norm(sphere_center - closest_point) <= sphere_radius


# --- 3. Main Calculation Function (Updated Logic) ---

def calculate_occlusion_with_threshold(threshold_percent):
    """Calculates total occlusion time based on a percentage threshold of blocked sight lines."""
    
    print(f"Starting calculation with threshold={threshold_percent}%...")
    print(f"Parameters: {NUM_TARGET_POINTS} target points, dt={TIME_STEP}s")
    start_time = time.time()
    
    # 1. Calculate the required number of blocked lines
    target_points = generate_target_points(NUM_TARGET_POINTS, true_target_base_center, true_target_radius, true_target_height)
    # Use ceiling to ensure that even a fraction requires the next whole number of lines.
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * NUM_TARGET_POINTS))
    print(f"Condition: At least {required_occluded_count} out of {NUM_TARGET_POINTS} lines must be blocked.")

    # 2. Pre-calculate trajectories (same as before)
    direction_vector = false_target_pos - uav_initial_pos
    direction_vector[2] = 0
    uav_velocity_vector = (direction_vector / np.linalg.norm(direction_vector)) * UAV_SPEED
    missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * MISSILE_SPEED
    explode_time = DROP_TIME + EXPLODE_DELAY
    drop_pos = uav_initial_pos + uav_velocity_vector * DROP_TIME
    drop_velocity = uav_velocity_vector
    explode_pos = drop_pos + drop_velocity * (explode_time - DROP_TIME) + np.array([0, 0, -0.5 * GRAVITY * (explode_time - DROP_TIME)**2])
    
    # 3. Iterate through time and check for occlusion
    total_occluded_time = 0.0
    simulation_start_time = explode_time
    simulation_end_time = explode_time + SMOKE_DURATION
    
    for t in np.arange(simulation_start_time, simulation_end_time, TIME_STEP):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        smoke_center_pos = explode_pos + np.array([0, 0, -3.0 * (t - explode_time)])
        
        # --- MODIFIED LOGIC ---
        # Count the number of occluded lines instead of stopping at the first one.
        occluded_lines_this_step = 0
        for target_point in target_points:
            if is_line_segment_intersecting_sphere(missile_pos, target_point, smoke_center_pos, SMOKE_RADIUS):
                occluded_lines_this_step += 1
        
        # Check if the count meets the threshold
        if occluded_lines_this_step >= required_occluded_count:
            total_occluded_time += TIME_STEP
            
    end_time = time.time()
    print(f"Calculation finished in {end_time - start_time:.2f} seconds.")
    
    return total_occluded_time


# --- 4. Run the Calculation and Print Result ---

if __name__ == "__main__":
    final_occlusion_time = calculate_occlusion_with_threshold(OCCLUSION_THRESHOLD_PERCENT)
    print("\n" + "="*50)
    print(f"  PRECISE OCCLUSION TIME (WITH THRESHOLD)")
    print("="*50)
    print(f"  Occlusion Threshold:         {OCCLUSION_THRESHOLD_PERCENT}%")
    print(f"  Effective Occlusion Time:    {final_occlusion_time:.4f} seconds")
    print("="*50)