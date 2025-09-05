import plotly.graph_objects as go
import numpy as np

def visualize_strategy(v, theta_deg, t_drop, t_delay):
    """
    根据给定的策略参数，生成并显示战场态势的动态模拟动画。

    参数:
    v (float): 无人机飞行速度 (m/s)。
    theta_deg (float): 无人机水平飞行方向 (度)。0度代表X轴正方向，90度代表Y轴正方向。
    t_drop (float): 从任务开始到投放干扰弹的时间 (秒)。
    t_delay (float): 从投放到干扰弹起爆的延迟时间 (秒)。
    """
    print("--- 开始生成策略可视化动画 ---")
    print(f"输入策略: V={v:.2f} m/s, θ={theta_deg:.2f}°, T_drop={t_drop:.2f}s, T_delay={t_delay:.2f}s")

    # --- 1. 模拟参数和场景设置 ---
    SIM_DURATION = 80  # 适当延长模拟总时长以适应不同策略
    TIME_STEP = 0.1
    GRAVITY = 9.8
    SMOKE_DURATION = 20.0
    SMOKE_RADIUS = 10.0

    # -- 目标和威胁数据 --
    uav_initial_pos = np.array([17800, 0, 1800], dtype=float)
    missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
    false_target = np.array([0, 0, 0], dtype=float)
    missile_speed = 300.0
    true_target_base_center = np.array([0, 200, 0], dtype=float)
    true_target_radius = 7
    true_target_height = 10
    true_target_los_point = true_target_base_center + np.array([0, 0, true_target_height / 2])

    # --- 2. 基于输入参数计算轨迹 ---
    
    # 将角度转换为弧度
    theta_rad = np.radians(theta_deg)
    # 根据速度和方向计算无人机速度向量
    uav_velocity_vector = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
    
    explode_time = t_drop + t_delay
    
    # 轨迹预计算
    times = np.arange(0, SIM_DURATION, TIME_STEP)
    trajectory = {}
    
    missile_velocity_vector = (false_target - missile_initial_pos) / np.linalg.norm(false_target - missile_initial_pos) * missile_speed
    trajectory['M1'] = [missile_initial_pos + missile_velocity_vector * t for t in times]
    trajectory['FY1'] = [uav_initial_pos + uav_velocity_vector * t for t in times]
    
    drop_pos = uav_initial_pos + uav_velocity_vector * t_drop
    drop_velocity = uav_velocity_vector
    explode_pos = drop_pos + drop_velocity * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
    
    grenade_traj = []
    smoke_cloud_traj = []
    for t in times:
        if t_drop <= t < explode_time:
            dt = t - t_drop
            grenade_traj.append(drop_pos + drop_velocity * dt + np.array([0, 0, -0.5 * GRAVITY * dt**2]))
        else:
            grenade_traj.append(np.full(3, np.nan))
            
        if explode_time <= t < explode_time + SMOKE_DURATION:
            dt_sink = t - explode_time
            smoke_cloud_traj.append(explode_pos + np.array([0, 0, -3.0 * dt_sink]))
        else:
            smoke_cloud_traj.append(np.full(3, np.nan))
            
    trajectory['Grenade'] = grenade_traj
    trajectory['SmokeCloud'] = smoke_cloud_traj

    # --- 3. 遮蔽判断与计时 ---
    def is_occluded(missile_pos, smoke_center_pos, target_pos, smoke_radius):
        if np.any(np.isnan(smoke_center_pos)): return False
        line_vec, point_vec = target_pos - missile_pos, smoke_center_pos - missile_pos
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0.0: return np.linalg.norm(point_vec) <= smoke_radius
        t = np.dot(point_vec, line_vec) / line_len_sq
        closest_point = missile_pos + np.clip(t, 0, 1) * line_vec
        return np.linalg.norm(smoke_center_pos - closest_point) <= smoke_radius

    occlusion_status = []
    cumulative_occlusion_time = []
    total_occluded_time = 0
    for i, t in enumerate(times):
        status = is_occluded(trajectory['M1'][i], trajectory['SmokeCloud'][i], true_target_los_point, SMOKE_RADIUS)
        occlusion_status.append(status)
        if status:
            total_occluded_time += TIME_STEP
        cumulative_occlusion_time.append(total_occluded_time)

    print(f"模拟计算完成。该策略的有效遮蔽总时长为: {total_occluded_time:.2f} 秒")

    # --- 4. 创建 Plotly 动画 (代码结构与之前相同) ---
    fig = go.Figure()
    
    # 静态轨迹
    fig.add_trace(go.Scatter3d(x=[u[0] for u in trajectory['FY1']], y=[u[1] for u in trajectory['FY1']], z=[u[2] for u in trajectory['FY1']], mode='lines', line=dict(color='blue', width=1, dash='dot'), name='FY1 轨迹'))
    fig.add_trace(go.Scatter3d(x=[m[0] for m in trajectory['M1']], y=[m[1] for m in trajectory['M1']], z=[m[2] for m in trajectory['M1']], mode='lines', line=dict(color='red', width=1, dash='dot'), name='M1 弹道'))
    
    # 动态物体初始位置
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0]], y=[trajectory['M1'][0][1]], z=[trajectory['M1'][0][2]], mode='markers', marker=dict(color='red', size=5), name='M1'))
    fig.add_trace(go.Scatter3d(x=[trajectory['FY1'][0][0]], y=[trajectory['FY1'][0][1]], z=[trajectory['FY1'][0][2]], mode='markers', marker=dict(color='blue', size=5), name='FY1'))
    fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='grey', size=4), name='烟幕弹'))
    fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='green', size=SMOKE_RADIUS, opacity=0.3), name='烟幕云团'))
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0], true_target_los_point[0]], y=[trajectory['M1'][0][1], true_target_los_point[1]], z=[trajectory['M1'][0][2], true_target_los_point[2]], mode='lines', line=dict(color='lime', width=4), name='视线 (LOS)'))

    # 静态目标
    fig.add_trace(go.Scatter3d(x=[false_target[0]], y=[false_target[1]], z=[false_target[2]], mode='markers', marker=dict(color='black', symbol='x', size=8), name='假目标'))
    u_cyl, h_cyl = np.linspace(0, 2*np.pi, 50), np.linspace(0, true_target_height, 2)
    u, h = np.meshgrid(u_cyl, h_cyl)
    x_cyl, y_cyl, z_cyl = true_target_base_center[0] + true_target_radius*np.cos(u), true_target_base_center[1] + true_target_radius*np.sin(u), true_target_base_center[2] + h
    fig.add_trace(go.Surface(x=x_cyl, y=y_cyl, z=z_cyl, colorscale='Greens', showscale=False, opacity=0.5, name='真目标'))
    
    # 创建帧
    frames = []
    for i, t in enumerate(times):
        los_color = 'magenta' if occlusion_status[i] else 'lime'
        status_text = "是" if occlusion_status[i] else "否"
        frame = go.Frame(
            data=[
                go.Scatter3d(x=[trajectory['M1'][i][0]], y=[trajectory['M1'][i][1]], z=[trajectory['M1'][i][2]]),
                go.Scatter3d(x=[trajectory['FY1'][i][0]], y=[trajectory['FY1'][i][1]], z=[trajectory['FY1'][i][2]]),
                go.Scatter3d(x=[trajectory['Grenade'][i][0]], y=[trajectory['Grenade'][i][1]], z=[trajectory['Grenade'][i][2]]),
                go.Scatter3d(x=[trajectory['SmokeCloud'][i][0]], y=[trajectory['SmokeCloud'][i][1]], z=[trajectory['SmokeCloud'][i][2]]),
                go.Scatter3d(x=[trajectory['M1'][i][0], true_target_los_point[0]], y=[trajectory['M1'][i][1], true_target_los_point[1]], z=[trajectory['M1'][i][2], true_target_los_point[2]], line=dict(color=los_color))
            ],
            name=f"t={t:.1f}s",
            traces=[2, 3, 4, 5, 6],
            layout={'title': f"战场模拟 | 时间: {t:.1f}s | 是否遮蔽: <b style='color:{los_color};'>{status_text}</b> | 累计遮蔽: {cumulative_occlusion_time[i]:.2f}s"}
        )
        frames.append(frame)
    fig.frames = frames

    # 布局和播放器
    fig.update_layout(
        title=f"战场模拟 | 时间: 0.0s | 是否遮蔽: 否 | 累计遮蔽: 0.00s",
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
    
    # 示例1：使用问题一的参数
    strategy_v = 120.0
    strategy_theta_deg = 180.0  # 飞向假目标，即-X方向
    strategy_t_drop = 1.5
    strategy_t_delay = 3.6
    
    # 示例2：使用你从第二问优化代码中得到的最优解
    # (请将下面替换为你的实际计算结果)
    # strategy_v = 77.4556 
    # strategy_theta_deg = 178.2870
    # strategy_t_drop = 0.3455 
    # strategy_t_delay = 2.6846 

    # 调用函数，生成并显示动画
    visualize_strategy(
        v=strategy_v,
        theta_deg=strategy_theta_deg,
        t_drop=strategy_t_drop,
        t_delay=strategy_t_delay
    )