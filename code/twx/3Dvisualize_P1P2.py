import plotly.graph_objects as go
import numpy as np
from numba import njit # [新增] 导入numba以进行计算加速

# ==============================================================================
# 1. [新增] Numba 加速的核心数学函数
# ==============================================================================
@njit(fastmath=True)
def is_line_segment_intersecting_sphere_numba(p1, p2, sphere_center, sphere_radius):
    """[Numba JIT加速] 检查线段p1-p2是否与球体相交"""
    if np.any(np.isnan(sphere_center)): return False
    line_vec = p2 - p1
    point_vec = sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.sqrt(np.sum(point_vec**2)) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0.0, min(1.0, t)) # 使用max/min以兼容numba
    closest_point_on_line = p1 + t * line_vec
    dist_sq = np.sum((sphere_center - closest_point_on_line)**2)
    return dist_sq <= sphere_radius**2

# ==============================================================================
# 2. 主可视化函数
# ==============================================================================
def visualize_strategy(v, theta_deg, t_drop, t_delay, n_points=1000, threshold_percent=70.0):
    """
    [核心修改] 根据给定策略，使用多点采样和百分比阈值生成战场态势的动态模拟动画。
    """
    print("--- 开始生成高精度策略可视化动画 ---")
    print(f"输入策略: V={v:.2f} m/s, θ={theta_deg:.2f}°, T_drop={t_drop:.2f}s, T_delay={t_delay:.2f}s")
    print(f"验证标准: {n_points} 个采样点, {threshold_percent}% 遮蔽阈值")

    # --- 1. 模拟参数和场景设置 ---
    SIM_DURATION = 80; TIME_STEP = 0.1; GRAVITY = 9.8
    SMOKE_DURATION = 20.0; SMOKE_RADIUS = 10.0
    uav_initial_pos = np.array([17800, 0, 1800], dtype=float)
    missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
    false_target = np.array([0, 0, 0], dtype=float)
    missile_speed = 300.0
    true_target_base_center = np.array([0, 200, 0], dtype=float)
    true_target_radius = 7; true_target_height = 10

    # --- 2. [新增] 生成目标表面采样点 ---
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points))
    target_points = []
    # 60%的点分布在圆柱侧面
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), true_target_height * np.random.rand()
        target_points.append([true_target_base_center[0] + true_target_radius * np.cos(theta), true_target_base_center[1] + true_target_radius * np.sin(theta), z])
    # 40%的点分布在上下底面
    for _ in range(int(n_points * 0.4)):
        r, theta = true_target_radius * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        target_points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), z])
    target_points = np.array(target_points)

    # --- 3. 基于输入参数计算轨迹 (逻辑不变) ---
    theta_rad = np.radians(theta_deg)
    uav_velocity_vector = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
    explode_time = t_drop + t_delay
    times = np.arange(0, SIM_DURATION, TIME_STEP)
    trajectory = {}
    missile_velocity_vector = (false_target - missile_initial_pos) / np.linalg.norm(false_target - missile_initial_pos) * missile_speed
    trajectory['M1'] = [missile_initial_pos + missile_velocity_vector * t for t in times]
    trajectory['FY1'] = [uav_initial_pos + uav_velocity_vector * t for t in times]
    drop_pos = uav_initial_pos + uav_velocity_vector * t_drop
    explode_pos = drop_pos + uav_velocity_vector * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
    grenade_traj, smoke_cloud_traj = [], []
    for t in times:
        if t_drop <= t < explode_time:
            dt = t - t_drop; grenade_traj.append(drop_pos + uav_velocity_vector * dt + np.array([0, 0, -0.5 * GRAVITY * dt**2]))
        else: grenade_traj.append(np.full(3, np.nan))
        if explode_time <= t < explode_time + SMOKE_DURATION:
            dt_sink = t - explode_time; smoke_cloud_traj.append(explode_pos + np.array([0, 0, -3.0 * dt_sink]))
        else: smoke_cloud_traj.append(np.full(3, np.nan))
    trajectory['Grenade'] = grenade_traj
    trajectory['SmokeCloud'] = smoke_cloud_traj

    # --- 4. [核心修改] 高精度遮蔽判断与计时 ---
    occlusion_status = []; cumulative_occlusion_time = []; total_occluded_time = 0
    all_occluded_indices = [] # 存储每一步被遮蔽的视线索引
    for i, t in enumerate(times):
        missile_pos = trajectory['M1'][i]; smoke_pos = trajectory['SmokeCloud'][i]
        occluded_points_count = 0
        current_occluded_indices = []

        if not np.any(np.isnan(smoke_pos)):
            for point_idx, tp in enumerate(target_points):
                if is_line_segment_intersecting_sphere_numba(missile_pos, tp, smoke_pos, SMOKE_RADIUS):
                    occluded_points_count += 1
                    current_occluded_indices.append(point_idx)
        
        is_occluded_this_step = occluded_points_count >= required_occluded_count
        occlusion_status.append(is_occluded_this_step)
        all_occluded_indices.append(current_occluded_indices)

        if is_occluded_this_step: total_occluded_time += TIME_STEP
        cumulative_occlusion_time.append(total_occluded_time)

    print(f"模拟计算完成。该策略的有效遮蔽总时长为: {total_occluded_time:.2f} 秒")

    # --- 5. 创建 Plotly 动画 ---
    fig = go.Figure()
    
    # 静态轨迹
    fig.add_trace(go.Scatter3d(x=[u[0] for u in trajectory['FY1']], y=[u[1] for u in trajectory['FY1']], z=[u[2] for u in trajectory['FY1']], mode='lines', line=dict(color='blue', width=1, dash='dot'), name='FY1 轨迹'))
    fig.add_trace(go.Scatter3d(x=[m[0] for m in trajectory['M1']], y=[m[1] for m in trajectory['M1']], z=[m[2] for m in trajectory['M1']], mode='lines', line=dict(color='red', width=1, dash='dot'), name='M1 弹道'))
    
    # 动态物体初始位置
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0]], y=[trajectory['M1'][0][1]], z=[trajectory['M1'][0][2]], mode='markers', marker=dict(color='red', size=5), name='M1'))
    fig.add_trace(go.Scatter3d(x=[trajectory['FY1'][0][0]], y=[trajectory['FY1'][0][1]], z=[trajectory['FY1'][0][2]], mode='markers', marker=dict(color='blue', size=5), name='FY1'))
    fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='grey', size=4), name='烟幕弹'))
    fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='green', size=SMOKE_RADIUS, opacity=0.3), name='烟幕云团'))
    
    # [新增] 用于显示被遮蔽视线的轨迹
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='magenta', width=1), name='被遮蔽视线'))
    # 主视线轨迹
    true_target_center_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0], true_target_center_point[0]], y=[trajectory['M1'][0][1], true_target_center_point[1]], z=[trajectory['M1'][0][2], true_target_center_point[2]], mode='lines', line=dict(color='lime', width=4), name='主视线 (LOS)'))

    # 静态目标
    fig.add_trace(go.Scatter3d(x=[false_target[0]], y=[false_target[1]], z=[false_target[2]], mode='markers', marker=dict(color='black', symbol='x', size=8), name='假目标'))
    u_cyl, h_cyl = np.linspace(0, 2*np.pi, 50), np.linspace(0, true_target_height, 2)
    u, h = np.meshgrid(u_cyl, h_cyl)
    x_cyl, y_cyl, z_cyl = true_target_base_center[0] + true_target_radius*np.cos(u), true_target_base_center[1] + true_target_radius*np.sin(u), true_target_base_center[2] + h
    fig.add_trace(go.Surface(x=x_cyl, y=y_cyl, z=z_cyl, colorscale='Greens', showscale=False, opacity=0.5, name='真目标'))
    # [新增] 可视化采样点
    fig.add_trace(go.Scatter3d(x=target_points[:, 0], y=target_points[:, 1], z=target_points[:, 2], mode='markers', marker=dict(color='darkgreen', size=1.5), name='目标采样点'))
    
    # 创建帧
    frames = []
    for i, t in enumerate(times):
        los_color = 'magenta' if occlusion_status[i] else 'lime'
        status_text = "是" if occlusion_status[i] else "否"
        
        # [新增] 构造被遮蔽视线的坐标数据
        occluded_los_lines_x, occluded_los_lines_y, occluded_los_lines_z = [], [], []
        if occlusion_status[i]:
            # 为了画面清晰，只画最多10条被遮蔽的线
            for point_idx in all_occluded_indices[i][:10]:
                tp = target_points[point_idx]
                occluded_los_lines_x.extend([trajectory['M1'][i][0], tp[0], None])
                occluded_los_lines_y.extend([trajectory['M1'][i][1], tp[1], None])
                occluded_los_lines_z.extend([trajectory['M1'][i][2], tp[2], None])

        frame = go.Frame(
            data=[
                go.Scatter3d(x=[trajectory['M1'][i][0]], y=[trajectory['M1'][i][1]], z=[trajectory['M1'][i][2]]),
                go.Scatter3d(x=[trajectory['FY1'][i][0]], y=[trajectory['FY1'][i][1]], z=[trajectory['FY1'][i][2]]),
                go.Scatter3d(x=[trajectory['Grenade'][i][0]], y=[trajectory['Grenade'][i][1]], z=[trajectory['Grenade'][i][2]]),
                go.Scatter3d(x=[trajectory['SmokeCloud'][i][0]], y=[trajectory['SmokeCloud'][i][1]], z=[trajectory['SmokeCloud'][i][2]]),
                go.Scatter3d(x=occluded_los_lines_x, y=occluded_los_lines_y, z=occluded_los_lines_z), # 更新被遮蔽视线
                go.Scatter3d(x=[trajectory['M1'][i][0], true_target_center_point[0]], y=[trajectory['M1'][i][1], true_target_center_point[1]], z=[trajectory['M1'][i][2], true_target_center_point[2]], line=dict(color=los_color)) # 更新主视线
            ],
            name=f"t={t:.1f}s",
            traces=[2, 3, 4, 5, 6, 7], # 更新 M1, FY1, Grenade, Smoke, OccludedLOS, MainLOS
            layout={'title': f"战场模拟 | 时间: {t:.1f}s | 是否遮蔽({threshold_percent}%): <b style='color:{los_color};'>{status_text}</b> | 累计遮蔽: {cumulative_occlusion_time[i]:.2f}s"}
        )
        frames.append(frame)
    fig.frames = frames

    # 布局和播放器
    fig.update_layout(
        title=f"战场模拟 | 时间: 0.0s | 是否遮蔽({threshold_percent}%): 否 | 累计遮蔽: 0.00s",
        scene=dict(
            xaxis=dict(title='X轴 (m)', range=[-1000, 21000]),
            yaxis=dict(title='Y轴 (m)', range=[-4000, 4000]),
            zaxis=dict(title='Z轴 (m)', range=[0, 2200]),
            aspectmode='manual', aspectratio=dict(x=5, y=2, z=0.8)
        ),
        updatemenus=[{'type': 'buttons','buttons': [
                {'label': '播放', 'method': 'animate', 'args': [None, {'frame': {'duration': 1000*TIME_STEP, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True}]},
                {'label': '暂停', 'method': 'animate', 'args': [[None], {'mode': 'immediate'}]}
            ]}],
        sliders=[{'steps': [
                {'label': f"{t:.1f}s", 'method': 'animate', 'args': [[f"t={t:.1f}s"], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]} for t in times
            ],'transition': {'duration': 0},'x': 0.1, 'len': 0.9}]
    )
    
    fig.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    # ===================================================================
    # == 在这里修改你想要可视化的策略参数 ==
    # ===================================================================
    
    # 示例1：问题一的参数
    # strategy_v = 120.0
    # strategy_theta_deg = 180.0
    # strategy_t_drop = 1.5
    # strategy_t_delay = 3.6
    
    # 示例2：问题二的某个优化解 (请替换为你的最优解)
    strategy_v = 77.4556 
    strategy_theta_deg = 178.2870
    strategy_t_drop = 0.3455 
    strategy_t_delay = 2.6846 

    # 调用函数，生成并显示动画
    visualize_strategy(
        v=strategy_v,
        theta_deg=strategy_theta_deg,
        t_drop=strategy_t_drop,
        t_delay=strategy_t_delay,
        n_points=1000, # 使用1000个采样点
        threshold_percent=70.0 # 70%的遮蔽阈值
    )