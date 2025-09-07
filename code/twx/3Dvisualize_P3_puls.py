import plotly.graph_objects as go
import numpy as np
from numba import njit

@njit(fastmath=True)
def is_line_segment_intersecting_sphere_numba(p1, p2, sphere_center, sphere_radius):
    if np.any(np.isnan(sphere_center)): return False
    line_vec = p2 - p1
    point_vec = sphere_center - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: return np.sqrt(np.sum(point_vec**2)) <= sphere_radius
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0.0, min(1.0, t))
    closest_point_on_line = p1 + t * line_vec
    dist_sq = np.sum((sphere_center - closest_point_on_line)**2)
    return dist_sq <= sphere_radius**2

def visualize_multi_grenade_strategy(v, theta_deg, drops, n_points=1000, threshold_percent=70.0):
    num_grenades = len(drops)
    print(f"--- 开始生成 {num_grenades} 枚烟幕弹协同策略的高精度可视化动画 ---")
    print(f"--- 验证标准: {n_points} 个采样点, {threshold_percent}% 遮蔽阈值 ---")
    print(f"输入策略: V={v:.2f} m/s, θ={theta_deg:.2f}°")

    SIM_DURATION = 80; TIME_STEP = 0.001; GRAVITY = 9.8
    SMOKE_DURATION = 20.0; SMOKE_RADIUS = 10.0
    uav_initial_pos = np.array([17800, 0, 1800], dtype=float)
    missile_initial_pos = np.array([20000, 0, 2000], dtype=float)
    false_target = np.array([0, 0, 0], dtype=float)
    missile_speed = 300.0
    true_target_base_center = np.array([0, 200, 0], dtype=float)
    true_target_radius = 7; true_target_height = 10
    
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points))
    target_points = []
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), true_target_height * np.random.rand()
        target_points.append([true_target_base_center[0] + true_target_radius * np.cos(theta), true_target_base_center[1] + true_target_radius * np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = true_target_radius * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else true_target_height
        target_points.append([true_target_base_center[0] + r * np.cos(theta), true_target_base_center[1] + r * np.sin(theta), z])
    target_points = np.array(target_points)

    theta_rad = np.radians(theta_deg)
    uav_velocity_vector = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
    times = np.arange(0, SIM_DURATION, TIME_STEP)
    trajectory = {}
    missile_velocity_vector = (false_target - missile_initial_pos) / np.linalg.norm(false_target - missile_initial_pos) * missile_speed
    trajectory['M1'] = [missile_initial_pos + missile_velocity_vector * t for t in times]
    trajectory['FY1'] = [uav_initial_pos + uav_velocity_vector * t for t in times]
    grenade_trajs, smoke_cloud_trajs = [], []
    for t_drop, t_delay in drops:
        explode_time = t_drop + t_delay
        drop_pos = uav_initial_pos + uav_velocity_vector * t_drop
        explode_pos = drop_pos + uav_velocity_vector * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
        g_traj, s_traj = [], []
        for t in times:
            if t_drop <= t < explode_time:
                g_traj.append(drop_pos + uav_velocity_vector * (t-t_drop) + np.array([0, 0, -0.5 * GRAVITY * (t-t_drop)**2]))
            else: g_traj.append(np.full(3, np.nan))
            if explode_time <= t < explode_time + SMOKE_DURATION:
                s_traj.append(explode_pos + np.array([0, 0, -3.0 * (t-explode_time)]))
            else: s_traj.append(np.full(3, np.nan))
        grenade_trajs.append(g_traj)
        smoke_cloud_trajs.append(s_traj)

    cooperative_occlusion_status = []
    cumulative_cooperative_time = []; total_cooperative_time = 0
    individual_occlusion_times = [0.0] * num_grenades
    cumulative_individual_times_history = []
    all_occluded_indices_history = []

    for i, t in enumerate(times):
        missile_pos = trajectory['M1'][i]
        total_occluded_indices_this_step = set()
        
        for j in range(num_grenades):
            smoke_pos = smoke_cloud_trajs[j][i]
            occluded_count_for_this_grenade = 0
            if not np.any(np.isnan(smoke_pos)):
                for point_idx, tp in enumerate(target_points):
                    if is_line_segment_intersecting_sphere_numba(missile_pos, tp, smoke_pos, SMOKE_RADIUS):
                        occluded_count_for_this_grenade += 1
                        total_occluded_indices_this_step.add(point_idx)
            
            if occluded_count_for_this_grenade >= required_occluded_count:
                individual_occlusion_times[j] += TIME_STEP

        is_coop_occluded = len(total_occluded_indices_this_step) >= required_occluded_count
        cooperative_occlusion_status.append(is_coop_occluded)
        all_occluded_indices_history.append(total_occluded_indices_this_step)
        
        if is_coop_occluded: total_cooperative_time += TIME_STEP
        cumulative_cooperative_time.append(total_cooperative_time)
        cumulative_individual_times_history.append(list(individual_occlusion_times))

    print(f"\n模拟计算完成。协同策略的总有效遮蔽时长为: {total_cooperative_time:.2f} 秒")
    print("--- 各烟幕弹独立贡献时长 (高精度) ---")
    for i, individual_time in enumerate(individual_occlusion_times):
        print(f"  烟幕弹 {i+1}: {individual_time:.2f} 秒")

    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(x=[u[0] for u in trajectory['FY1']], y=[u[1] for u in trajectory['FY1']], z=[u[2] for u in trajectory['FY1']], mode='lines', line=dict(color='blue', width=1, dash='dot'), name='FY1 轨迹'))
    fig.add_trace(go.Scatter3d(x=[m[0] for m in trajectory['M1']], y=[m[1] for m in trajectory['M1']], z=[m[2] for m in trajectory['M1']], mode='lines', line=dict(color='red', width=1, dash='dot'), name='M1 弹道'))
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0]], y=[trajectory['M1'][0][1]], z=[trajectory['M1'][0][2]], mode='markers', marker=dict(color='red', size=5), name='M1'))
    fig.add_trace(go.Scatter3d(x=[trajectory['FY1'][0][0]], y=[trajectory['FY1'][0][1]], z=[trajectory['FY1'][0][2]], mode='markers', marker=dict(color='blue', size=5), name='FY1'))
    grenade_colors = ['grey', 'dimgray', 'darkgrey']; smoke_colors = ['green', 'seagreen', 'lightgreen']
    for i in range(num_grenades):
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color=grenade_colors[i], size=4), name=f'烟幕弹 {i+1}'))
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color=smoke_colors[i], size=SMOKE_RADIUS, opacity=0.3), name=f'烟幕云团 {i+1}'))
    
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='magenta', width=1), name='被遮蔽视线'))
    true_target_center_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0], true_target_center_point[0]], y=[trajectory['M1'][0][1], true_target_center_point[1]], z=[trajectory['M1'][0][2], true_target_center_point[2]], mode='lines', line=dict(color='lime', width=4), name='主视线 (LOS)'))
    
    fig.add_trace(go.Scatter3d(x=[false_target[0]], y=[false_target[1]], z=[false_target[2]], mode='markers', marker=dict(color='black', symbol='x', size=8), name='假目标'))
    u_cyl, h_cyl = np.linspace(0, 2*np.pi, 50), np.linspace(0, true_target_height, 2); u, h = np.meshgrid(u_cyl, h_cyl)
    x_cyl, y_cyl, z_cyl = true_target_base_center[0] + true_target_radius*np.cos(u), true_target_base_center[1] + true_target_radius*np.sin(u), true_target_base_center[2] + h
    fig.add_trace(go.Surface(x=x_cyl, y=y_cyl, z=z_cyl, colorscale='Greens', showscale=False, opacity=0.5, name='真目标'))
    fig.add_trace(go.Scatter3d(x=target_points[:, 0], y=target_points[:, 1], z=target_points[:, 2], mode='markers', marker=dict(color='darkgreen', size=1.5), name='目标采样点'))
    
    frames = []
    for i, t in enumerate(times):
        frame_data = [
            go.Scatter3d(x=[trajectory['M1'][i][0]], y=[trajectory['M1'][i][1]], z=[trajectory['M1'][i][2]]),
            go.Scatter3d(x=[trajectory['FY1'][i][0]], y=[trajectory['FY1'][i][1]], z=[trajectory['FY1'][i][2]])
        ]
        for j in range(num_grenades):
            frame_data.append(go.Scatter3d(x=[grenade_trajs[j][i][0]], y=[grenade_trajs[j][i][1]], z=[grenade_trajs[j][i][2]]))
            frame_data.append(go.Scatter3d(x=[smoke_cloud_trajs[j][i][0]], y=[smoke_cloud_trajs[j][i][1]], z=[smoke_cloud_trajs[j][i][2]]))
        
        occluded_los_lines_x, occluded_los_lines_y, occluded_los_lines_z = [], [], []
        if cooperative_occlusion_status[i]:
            for point_idx in list(all_occluded_indices_history[i])[:10]:
                tp = target_points[point_idx]
                occluded_los_lines_x.extend([trajectory['M1'][i][0], tp[0], None])
                occluded_los_lines_y.extend([trajectory['M1'][i][1], tp[1], None])
                occluded_los_lines_z.extend([trajectory['M1'][i][2], tp[2], None])
        frame_data.append(go.Scatter3d(x=occluded_los_lines_x, y=occluded_los_lines_y, z=occluded_los_lines_z))
        
        los_color = 'magenta' if cooperative_occlusion_status[i] else 'lime'
        status_text = "是" if cooperative_occlusion_status[i] else "否"
        frame_data.append(go.Scatter3d(x=[trajectory['M1'][i][0], true_target_center_point[0]], y=[trajectory['M1'][i][1], true_target_center_point[1]], z=[trajectory['M1'][i][2], true_target_center_point[2]], line=dict(color=los_color)))
        
        individual_times_str = " | ".join([f"弹{k+1}: {cumulative_individual_times_history[i][k]:.2f}s" for k in range(num_grenades)])
        title = (f"战场模拟 | 时间: {t:.1f}s | 协同遮蔽({threshold_percent}%): <b style='color:{los_color};'>{status_text}</b>"
                 f"<br>累计(协同): {cumulative_cooperative_time[i]:.2f}s | 累计(独立): {individual_times_str}")
        
        frame = go.Frame(data=frame_data, name=f"t={t:.1f}s", traces=list(range(2, 2 + 2*num_grenades + 2)), layout={'title': title})
        frames.append(frame)
    fig.frames = frames

    initial_individual_str = " | ".join([f"弹{k+1}: 0.00s" for k in range(num_grenades)])
    initial_title = f"战场模拟 | 时间: 0.0s | 协同遮蔽({threshold_percent}%): 否<br>累计(协同): 0.00s | 累计(独立): {initial_individual_str}"
    
    fig.update_layout(title=initial_title, scene=dict(xaxis=dict(title='X轴 (m)', range=[-1000, 21000]), yaxis=dict(title='Y轴 (m)', range=[-4000, 4000]), zaxis=dict(title='Z轴 (m)', range=[0, 2200]), aspectmode='manual', aspectratio=dict(x=5, y=2, z=0.8)), updatemenus=[{'type': 'buttons','buttons': [{'label': '播放', 'method': 'animate', 'args': [None, {'frame': {'duration': 1000*TIME_STEP, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True}]}, {'label': '暂停', 'method': 'animate', 'args': [[None], {'mode': 'immediate'}]}]}], sliders=[{'steps': [{'label': f"{t:.1f}s", 'method': 'animate', 'args': [[f"t={t:.1f}s"], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]} for t in times],'transition': {'duration': 0},'x': 0.1, 'len': 0.9}])
    
    fig.show()


if __name__ == "__main__":
    
    strategy_v = 110.8789   
    strategy_theta_deg = 179.4645  
    
    strategy_drops = [
        (0.6146, 3.6631),   
        (3.4152, 4.6631),  
        (4.7752, 5.0631)   
    ]
    
    visualize_multi_grenade_strategy(
        v=strategy_v,
        theta_deg=strategy_theta_deg,
        drops=strategy_drops,
        n_points=1000,           
        threshold_percent=70.0   
    )