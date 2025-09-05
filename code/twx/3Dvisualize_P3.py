import plotly.graph_objects as go
import numpy as np

def visualize_multi_grenade_strategy(v, theta_deg, drops):
    """
    根据给定的多烟幕弹策略，生成并显示战场态势的动态模拟动画。

    参数:
    v (float): 无人机飞行速度 (m/s)。
    theta_deg (float): 无人机水平飞行方向 (度)。
    drops (list of tuples): 一个包含多枚烟幕弹策略的列表。
                              每个元组为 (t_drop, t_delay)。
                              例如: [(drop1, delay1), (drop2, delay2), (drop3, delay3)]
    """
    num_grenades = len(drops)
    print(f"--- 开始生成 {num_grenades} 枚烟幕弹协同策略的可视化动画 ---")
    print(f"输入策略: V={v:.2f} m/s, θ={theta_deg:.2f}°")
    for i, (t_drop, t_delay) in enumerate(drops):
        print(f"  弹药 {i+1}: T_drop={t_drop:.2f}s, T_delay={t_delay:.2f}s")

    # --- 1. 模拟参数和场景设置 ---
    SIM_DURATION = 80
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

    # --- 2. 基于输入参数计算所有轨迹 ---
    theta_rad = np.radians(theta_deg)
    uav_velocity_vector = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
    
    times = np.arange(0, SIM_DURATION, TIME_STEP)
    trajectory = {}
    
    missile_velocity_vector = (false_target - missile_initial_pos) / np.linalg.norm(false_target - missile_initial_pos) * missile_speed
    trajectory['M1'] = [missile_initial_pos + missile_velocity_vector * t for t in times]
    trajectory['FY1'] = [uav_initial_pos + uav_velocity_vector * t for t in times]
    
    # 为每枚烟幕弹计算轨迹
    grenade_trajs = []
    smoke_cloud_trajs = []
    for t_drop, t_delay in drops:
        explode_time = t_drop + t_delay
        drop_pos = uav_initial_pos + uav_velocity_vector * t_drop
        explode_pos = drop_pos + uav_velocity_vector * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        
        g_traj, s_traj = [], []
        for t in times:
            if t_drop <= t < explode_time:
                dt = t - t_drop
                g_traj.append(drop_pos + uav_velocity_vector * dt + np.array([0, 0, -0.5 * GRAVITY * dt**2]))
            else:
                g_traj.append(np.full(3, np.nan))
            
            if explode_time <= t < explode_time + SMOKE_DURATION:
                dt_sink = t - explode_time
                s_traj.append(explode_pos + np.array([0, 0, -3.0 * dt_sink]))
            else:
                s_traj.append(np.full(3, np.nan))
        grenade_trajs.append(g_traj)
        smoke_cloud_trajs.append(s_traj)

    # --- 3. 遮蔽判断与计时 (检查所有云团) ---
    def is_occluded(missile_pos, smoke_center_pos, target_pos, smoke_radius):
        # (函数本身不变)
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
        is_occluded_this_step = False
        # 检查是否被任何一个烟幕云团遮蔽
        for smoke_cloud_traj in smoke_cloud_trajs:
            if is_occluded(trajectory['M1'][i], smoke_cloud_traj[i], true_target_los_point, SMOKE_RADIUS):
                is_occluded_this_step = True
                break
        
        occlusion_status.append(is_occluded_this_step)
        if is_occluded_this_step:
            total_occluded_time += TIME_STEP
        cumulative_occlusion_time.append(total_occluded_time)

    print(f"模拟计算完成。该策略的总有效遮蔽时长为: {total_occluded_time:.2f} 秒")

    # --- 4. 创建 Plotly 动画 ---
    fig = go.Figure()
    
    # 静态轨迹
    fig.add_trace(go.Scatter3d(x=[u[0] for u in trajectory['FY1']], y=[u[1] for u in trajectory['FY1']], z=[u[2] for u in trajectory['FY1']], mode='lines', line=dict(color='blue', width=1, dash='dot'), name='FY1 轨迹'))
    fig.add_trace(go.Scatter3d(x=[m[0] for m in trajectory['M1']], y=[m[1] for m in trajectory['M1']], z=[m[2] for m in trajectory['M1']], mode='lines', line=dict(color='red', width=1, dash='dot'), name='M1 弹道'))
    
    # 动态物体初始位置
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0]], y=[trajectory['M1'][0][1]], z=[trajectory['M1'][0][2]], mode='markers', marker=dict(color='red', size=5), name='M1'))
    fig.add_trace(go.Scatter3d(x=[trajectory['FY1'][0][0]], y=[trajectory['FY1'][0][1]], z=[trajectory['FY1'][0][2]], mode='markers', marker=dict(color='blue', size=5), name='FY1'))
    
    # 添加多枚烟幕弹和云团的占位符
    grenade_colors = ['grey', 'dimgray', 'darkgrey']
    smoke_colors = ['green', 'seagreen', 'lightgreen']
    for i in range(num_grenades):
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color=grenade_colors[i % len(grenade_colors)], size=4), name=f'烟幕弹 {i+1}'))
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color=smoke_colors[i % len(smoke_colors)], size=SMOKE_RADIUS, opacity=0.3), name=f'烟幕云团 {i+1}'))

    # 视线
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0], true_target_los_point[0]], y=[trajectory['M1'][0][1], true_target_los_point[1]], z=[trajectory['M1'][0][2], true_target_los_point[2]], mode='lines', line=dict(color='lime', width=4), name='视线 (LOS)'))

    # 静态目标 (代码不变)
    fig.add_trace(go.Scatter3d(x=[false_target[0]], y=[false_target[1]], z=[false_target[2]], mode='markers', marker=dict(color='black', symbol='x', size=8), name='假目标'))
    u_cyl, h_cyl = np.linspace(0, 2*np.pi, 50), np.linspace(0, true_target_height, 2)
    u, h = np.meshgrid(u_cyl, h_cyl)
    x_cyl, y_cyl, z_cyl = true_target_base_center[0] + true_target_radius*np.cos(u), true_target_base_center[1] + true_target_radius*np.sin(u), true_target_base_center[2] + h
    fig.add_trace(go.Surface(x=x_cyl, y=y_cyl, z=z_cyl, colorscale='Greens', showscale=False, opacity=0.5, name='真目标'))
    
    # 创建帧
    frames = []
    for i, t in enumerate(times):
        frame_data = [
            go.Scatter3d(x=[trajectory['M1'][i][0]], y=[trajectory['M1'][i][1]], z=[trajectory['M1'][i][2]]),
            go.Scatter3d(x=[trajectory['FY1'][i][0]], y=[trajectory['FY1'][i][1]], z=[trajectory['FY1'][i][2]])
        ]
        # 添加所有烟幕弹和云团的当前位置
        for j in range(num_grenades):
            frame_data.append(go.Scatter3d(x=[grenade_trajs[j][i][0]], y=[grenade_trajs[j][i][1]], z=[grenade_trajs[j][i][2]]))
            frame_data.append(go.Scatter3d(x=[smoke_cloud_trajs[j][i][0]], y=[smoke_cloud_trajs[j][i][1]], z=[smoke_cloud_trajs[j][i][2]]))
        
        los_color = 'magenta' if occlusion_status[i] else 'lime'
        status_text = "是" if occlusion_status[i] else "否"
        frame_data.append(go.Scatter3d(x=[trajectory['M1'][i][0], true_target_los_point[0]], y=[trajectory['M1'][i][1], true_target_los_point[1]], z=[trajectory['M1'][i][2], true_target_los_point[2]], line=dict(color=los_color)))
        
        frame = go.Frame(
            data=frame_data,
            name=f"t={t:.1f}s",
            # 更新需要修改的轨迹索引
            traces=[2, 3, 4, 5, 6, 7, 8, 9, 10], 
            layout={'title': f"战场模拟 | 时间: {t:.1f}s | 是否遮蔽: <b style='color:{los_color};'>{status_text}</b> | 累计遮蔽: {cumulative_occlusion_time[i]:.2f}s"}
        )
        frames.append(frame)
    fig.frames = frames

    # 布局和播放器 (代码不变)
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
    # == 在这里修改你想要可视化的第三问策略参数 ==
    # ===================================================================
    
    # 示例：使用你从第三问优化代码中得到的最优解
    # (请将下面替换为你的实际计算结果)
    strategy_v = 72.2367   # 无人机统一飞行速度
    strategy_theta_deg = 178.2870 # 无人机统一飞行方向
    
    # 三枚烟幕弹各自的 (投放时间, 引信延迟)
    strategy_drops = [
        (0.1 , 2.8846),   # 第 1 枚
        (1.7228, 2.8846),   # 第 2 枚
        (2.7455, 2.6846)    # 第 3 枚
    ]

    # 调用函数，生成并显示动画
    visualize_multi_grenade_strategy(
        v=strategy_v,
        theta_deg=strategy_theta_deg,
        drops=strategy_drops
    )