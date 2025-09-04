import numpy as np
import cma
import time

# --- 1. 适应度函数 (我们的核心模拟器) ---
# 这部分代码与之前 "精确计算器" 的逻辑完全相同
# 我们将其封装成一个函数，输入决策变量，输出要优化的目标值

# -- 固定的物理和场景参数 --
GRAVITY = 9.8
MISSILE_SPEED = 300.0
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0
TIME_STEP = 0.001  # 在优化中，可以用稍大的步长以加速，最后用小步长验证
NUM_TARGET_POINTS = 100 # 优化时可适当减少点数以提速

# -- 初始位置和目标几何 --
uav_initial_pos = np.array([17800, 0, 1800])
missile_initial_pos = np.array([20000, 0, 2000])
false_target_pos = np.array([0, 0, 0])
true_target_base_center = np.array([0, 200, 0])
true_target_radius = 7
true_target_height = 10



# (Helper functions are the same as before)
def generate_target_points(n_points, base_center, radius, height):
    points = []
    n_side = int(n_points * 0.7)
    n_caps = int(n_points * 0.15)
    for i in range(n_side):
        theta, z = 2 * np.pi * (i / n_side), height * np.random.rand()
        points.append([base_center[0] + radius * np.cos(theta), base_center[1] + radius * np.sin(theta), z])
    for i in range(n_caps * 2):
        r, theta = radius * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if i < n_caps else height
        points.append([base_center[0] + r * np.cos(theta), base_center[1] + r * np.sin(theta), base_center[2] + z])
    return np.array(points)

    # 预先生成目标点，避免在循环中重复计算
target_points = generate_target_points(
    NUM_TARGET_POINTS, true_target_base_center, true_target_radius, true_target_height
)

def is_line_segment_intersecting_sphere(p1, p2, sphere_center, sphere_radius):
    if np.any(np.isnan(sphere_center)): return False
    line_vec, point_vec = p2 - p1, sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.linalg.norm(point_vec) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    closest_point = p1 + np.clip(t, 0, 1) * line_vec
    return np.linalg.norm(sphere_center - closest_point) <= sphere_radius

def fitness_function(x):
    """
    接受一个包含4个决策变量的列表 x，返回负的遮蔽时间。
    CMA-ES 默认是最小化，所以我们需要返回负值。
    x = [v, theta, t_drop, t_delay]
    """
    uav_speed, uav_theta, t_drop, t_delay = x

    # --- 根据决策变量计算轨迹 ---
    # 无人机速度向量
    uav_velocity_vector = np.array([
        uav_speed * np.cos(uav_theta),
        uav_speed * np.sin(uav_theta),
        0  # 等高飞行
    ])
    
    # 导弹速度向量
    missile_velocity_vector = (false_target_pos - missile_initial_pos) / np.linalg.norm(false_target_pos - missile_initial_pos) * MISSILE_SPEED
    
    # 烟幕弹关键点和时间
    explode_time = t_drop + t_delay
    drop_pos = uav_initial_pos + uav_velocity_vector * t_drop
    drop_velocity = uav_velocity_vector
    explode_pos = drop_pos + drop_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])

    # --- 迭代计算遮蔽时间 ---
    total_occluded_time = 0.0
    simulation_start_time = explode_time
    simulation_end_time = explode_time + SMOKE_DURATION
    
    for t in np.arange(simulation_start_time, simulation_end_time, TIME_STEP):
        missile_pos = missile_initial_pos + missile_velocity_vector * t
        smoke_center_pos = explode_pos + np.array([0, 0, -3.0 * (t - explode_time)])
        
        is_occluded_this_step = False
        for target_point in target_points:
            if is_line_segment_intersecting_sphere(missile_pos, target_point, smoke_center_pos, SMOKE_RADIUS):
                is_occluded_this_step = True
                break
        
        if is_occluded_this_step:
            total_occluded_time += TIME_STEP
            
    # CMA-ES默认最小化，所以返回负值
    return -total_occluded_time

# --- 2. CMA-ES 优化器设置与执行 ---

if __name__ == "__main__":
    # 决策变量的边界
    # [v, theta, t_drop, t_delay]
    bounds = [[70, (-1)*(1/6)* np.pi, 1, 1], [140, (1/6)* np.pi, 30, 10]]

    # 初始猜测值 (x0) - 一个合理的、但不一定最优的策略
    # 比如：以中等速度飞向导弹轨迹与真目标y坐标平面的交点
    # 这是一个启发式的好起点，可以加速收敛
    initial_guess = [105, np.pi, 25, 5] 
    sigma0 = 0.5  # 初始搜索步长

    # 设置CMA-ES
    opts = cma.CMAOptions()
    opts.set('bounds', bounds)
    opts.set('maxfevals', 1000) # 最大函数评估次数，增加可提高精度但耗时更长
    opts.set('tolfun', 1e-3)    # 函数值变化的容忍度
    
    print("开始使用 CMA-ES 进行优化...")
    start_time = time.time()
    
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, opts)
    
    # 运行优化循环
    es.optimize(fitness_function, verb_disp=100) # verb_disp 每100代打印一次日志
    
    end_time = time.time()
    print(f"\n优化完成，耗时: {end_time - start_time:.2f} 秒")
    
    # --- 3. 输出结果 ---
    best_solution = es.result.xbest
    best_fitness = -es.result.fbest  # 转回正的最大遮蔽时间

    # 计算最优策略下的投放点和起爆点
    v_opt, theta_opt, t_drop_opt, t_delay_opt = best_solution
    uav_vel_opt = np.array([v_opt * np.cos(theta_opt), v_opt * np.sin(theta_opt), 0])
    drop_point_opt = uav_initial_pos + uav_vel_opt * t_drop_opt
    explode_point_opt = drop_point_opt + uav_vel_opt * t_delay_opt + \
                       np.array([0, 0, -0.5 * GRAVITY * t_delay_opt**2])

    print("\n" + "="*50)
    print("            最优策略结果 (问题二)")
    print("="*50)
    print(f"  最大有效遮蔽时间: {best_fitness:.4f} 秒")
    print("\n  --- 最优决策变量 ---")
    print(f"  无人机飞行速度 (v):      {v_opt:.4f} m/s")
    print(f"  无人机飞行方向 (θ):      {np.rad2deg(theta_opt):.4f} 度")
    print(f"  干扰弹投放时间 (t_drop): {t_drop_opt:.4f} s")
    print(f"  干扰弹起爆延迟 (t_delay):{t_delay_opt:.4f} s")
    print("\n  --- 对应物理坐标 ---")
    print(f"  干扰弹投放点: {np.round(drop_point_opt, 2)}")
    print(f"  干扰弹起爆点: {np.round(explode_point_opt, 2)}")
    print("="*50)