import plotly.graph_objects as go
import numpy as np
from numba import njit 

@njit(fastmath=True)
def is_line_segment_intersecting_sphere_numba(p1, p2, sphere_center, sphere_radius):
    """[Numba JIT加速] 检查线段p1-p2是否与球体相交"""
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

def visualize_team_strategy(team_strategies, n_points=100, threshold_percent=70.0):
    num_uavs = len(team_strategies)
    total_grenades = sum(len(s['drops']) for s in team_strategies.values())
    print(f"--- 开始生成 {num_uavs} 架无人机, 共 {total_grenades} 枚烟幕弹的协同策略可视化动画 ---")
    print(f"--- 使用高精度遮蔽判断: {n_points} 个采样点, {threshold_percent}% 遮蔽阈值 ---")
   
    SIM_DURATION = 80; TIME_STEP = 0.1; GRAVITY = 9.8
    SMOKE_DURATION = 20.0; SMOKE_RADIUS = 10.0
    UAV_INITIAL_POS = {'FY1': np.array([17800, 0, 1800], dtype=float), 'FY2': np.array([12000, 1400, 1400], dtype=float)}
    missile_initial_pos = np.array([20000, 0, 2000], dtype=float); false_target = np.array([0, 0, 0], dtype=float)
    missile_speed = 300.0; true_target_base_center = np.array([0, 200, 0], dtype=float)
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
    
    times = np.arange(0, SIM_DURATION, TIME_STEP); trajectory = {}
    missile_velocity_vector = (false_target - missile_initial_pos) / np.linalg.norm(false_target - missile_initial_pos) * missile_speed
    trajectory['M1'] = [missile_initial_pos + missile_velocity_vector * t for t in times]
    all_grenade_trajs = []; all_smoke_cloud_trajs = []
    for uav_name, strategy in team_strategies.items():
        v = strategy['v']; theta_rad = np.radians(strategy['theta_deg']); uav_initial_pos = UAV_INITIAL_POS[uav_name]
        uav_velocity_vector = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
        trajectory[uav_name] = [uav_initial_pos + uav_velocity_vector * t for t in times]
        for t_drop, t_delay in strategy['drops']:
            explode_time = t_drop + t_delay; drop_pos = uav_initial_pos + uav_velocity_vector * t_drop
            explode_pos = drop_pos + uav_velocity_vector * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
            g_traj, s_traj = [], []
            for t in times:
                if t_drop <= t < explode_time:
                    dt = t - t_drop; g_traj.append(drop_pos + uav_velocity_vector * dt + np.array([0, 0, -0.5 * GRAVITY * dt**2]))
                else: g_traj.append(np.full(3, np.nan))
                if explode_time <= t < explode_time + SMOKE_DURATION:
                    dt_sink = t - explode_time; s_traj.append(explode_pos + np.array([0, 0, -3.0 * dt_sink]))
                else: s_traj.append(np.full(3, np.nan))
            all_grenade_trajs.append(g_traj); all_smoke_cloud_trajs.append(s_traj)

    occlusion_status = []; cumulative_occlusion_time = []; total_occluded_time = 0
    for i, t in enumerate(times):
        missile_pos = trajectory['M1'][i]
        occluded_points_indices = set() 
        
        for smoke_traj in all_smoke_cloud_trajs:
            smoke_pos = smoke_traj[i]
            if np.any(np.isnan(smoke_pos)): continue
            
            for point_idx, tp in enumerate(target_points):
                if point_idx in occluded_points_indices: continue 
                if is_line_segment_intersecting_sphere_numba(missile_pos, tp, smoke_pos, SMOKE_RADIUS):
                    occluded_points_indices.add(point_idx)

        is_occluded_this_step = len(occluded_points_indices) >= required_occluded_count
        
        occlusion_status.append(is_occluded_this_step)
        if is_occluded_this_step: total_occluded_time += TIME_STEP
        cumulative_occlusion_time.append(total_occluded_time)

    print(f"模拟计算完成。该协同策略的总有效遮蔽时长为: {total_occluded_time:.2f} 秒")

    fig = go.Figure()
    uav_colors = {'FY1': 'blue', 'FY2': 'cyan'}
    grenade_colors = {'FY1': ['grey', 'dimgray', 'darkgrey'], 'FY2': ['navy', 'midnightblue', 'darkslateblue']}
    smoke_colors = {'FY1': ['green', 'seagreen', 'lightgreen'], 'FY2': ['turquoise', 'paleturquoise', 'lightcyan']}

    fig.add_trace(go.Scatter3d(x=[m[0] for m in trajectory['M1']], y=[m[1] for m in trajectory['M1']], z=[m[2] for m in trajectory['M1']], mode='lines', line=dict(color='red', width=1, dash='dot'), name='M1 弹道'))
    for uav_name in team_strategies.keys():
        uav_traj = trajectory[uav_name]; fig.add_trace(go.Scatter3d(x=[u[0] for u in uav_traj], y=[u[1] for u in uav_traj], z=[u[2] for u in uav_traj], mode='lines', line=dict(color=uav_colors[uav_name], width=1, dash='dot'), name=f'{uav_name} 轨迹'))
    
    animated_traces_start_index = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0]], y=[trajectory['M1'][0][1]], z=[trajectory['M1'][0][2]], mode='markers', marker=dict(color='red', size=5), name='M1'))
    for uav_name in team_strategies.keys(): fig.add_trace(go.Scatter3d(x=[trajectory[uav_name][0][0]], y=[trajectory[uav_name][0][1]], z=[trajectory[uav_name][0][2]], mode='markers', marker=dict(color=uav_colors[uav_name], size=5), name=uav_name))
    
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='magenta', width=1), name='被遮蔽视线'))
    true_target_center_point = true_target_base_center + np.array([0, 0, true_target_height / 2])
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0], true_target_center_point[0]], y=[trajectory['M1'][0][1], true_target_center_point[1]], z=[trajectory['M1'][0][2], true_target_center_point[2]], mode='lines', line=dict(color='lime', width=4), name='主视线 (LOS)'))
    
    for uav_name, strategy in team_strategies.items():
        for i in range(len(strategy['drops'])):
            fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color=grenade_colors[uav_name][i], size=4), name=f'{uav_name} 烟幕弹 {i+1}'))
            fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color=smoke_colors[uav_name][i], size=SMOKE_RADIUS, opacity=0.3), name=f'{uav_name} 烟幕 {i+1}'))

    fig.add_trace(go.Scatter3d(x=[false_target[0]], y=[false_target[1]], z=[false_target[2]], mode='markers', marker=dict(color='black', symbol='x', size=8), name='假目标'))
    u_cyl, h_cyl = np.linspace(0, 2*np.pi, 50), np.linspace(0, true_target_height, 2); u, h = np.meshgrid(u_cyl, h_cyl)
    x_cyl, y_cyl, z_cyl = true_target_base_center[0] + true_target_radius*np.cos(u), true_target_base_center[1] + true_target_radius*np.sin(u), true_target_base_center[2] + h
    fig.add_trace(go.Surface(x=x_cyl, y=y_cyl, z=z_cyl, colorscale='Greens', showscale=False, opacity=0.5, name='真目标'))
    fig.add_trace(go.Scatter3d(x=target_points[:, 0], y=target_points[:, 1], z=target_points[:, 2], mode='markers', marker=dict(color='darkgreen', size=1.5), name='目标采样点'))
    
    num_dynamic_traces = 2 + 1 + 1 + 2*total_grenades 
    
    
    frames = []
    for i, t in enumerate(times):
        missile_pos = trajectory['M1'][i]
        
        
        los_color = 'magenta' if occlusion_status[i] else 'lime'
        status_text = "是" if occlusion_status[i] else "否"
        occluded_los_lines_x, occluded_los_lines_y, occluded_los_lines_z = [], [], []
        if occlusion_status[i]:
            for point_idx in list(occluded_points_indices)[:10]: 
                tp = target_points[point_idx]
                occluded_los_lines_x.extend([missile_pos[0], tp[0], None])
                occluded_los_lines_y.extend([missile_pos[1], tp[1], None])
                occluded_los_lines_z.extend([missile_pos[2], tp[2], None])

        frame_data = [
            go.Scatter3d(x=[missile_pos[0]], y=[missile_pos[1]], z=[missile_pos[2]]),
            go.Scatter3d(x=[trajectory['FY1'][i][0]], y=[trajectory['FY1'][i][1]], z=[trajectory['FY1'][i][2]]),
            go.Scatter3d(x=[trajectory['FY2'][i][0]], y=[trajectory['FY2'][i][1]], z=[trajectory['FY2'][i][2]]),
            go.Scatter3d(x=occluded_los_lines_x, y=occluded_los_lines_y, z=occluded_los_lines_z), 
            go.Scatter3d(x=[missile_pos[0], true_target_center_point[0]], y=[missile_pos[1], true_target_center_point[1]], z=[missile_pos[2], true_target_center_point[2]], line=dict(color=los_color)) 
        ]
        
        for j in range(total_grenades):
            frame_data.append(go.Scatter3d(x=[all_grenade_trajs[j][i][0]], y=[all_grenade_trajs[j][i][1]], z=[all_grenade_trajs[j][i][2]]))
            frame_data.append(go.Scatter3d(x=[all_smoke_cloud_trajs[j][i][0]], y=[all_smoke_cloud_trajs[j][i][1]], z=[all_smoke_cloud_trajs[j][i][2]]))
        
        traces_to_update = list(range(animated_traces_start_index, len(fig.data) - 3)) 
        
        frame = go.Frame(
            data=frame_data,
            name=f"t={t:.1f}s",
            traces=traces_to_update,
            layout={'title': f"战场模拟 | 时间: {t:.1f}s | 是否遮蔽({threshold_percent}%): <b style='color:{los_color};'>{status_text}</b> | 累计遮蔽: {cumulative_occlusion_time[i]:.2f}s"}
        )
        frames.append(frame)
    fig.frames = frames

    fig.update_layout(title=f"战场模拟 | 时间: 0.0s | 是否遮蔽({threshold_percent}%): 否 | 累计遮蔽: 0.00s", scene=dict(xaxis=dict(title='X轴 (m)', range=[-1000, 21000]), yaxis=dict(title='Y轴 (m)', range=[-4000, 4000]), zaxis=dict(title='Z轴 (m)', range=[0, 2200]), aspectmode='manual', aspectratio=dict(x=5, y=2, z=0.8)), updatemenus=[{'type': 'buttons','buttons': [{'label': '播放', 'method': 'animate', 'args': [None, {'frame': {'duration': 1000*TIME_STEP, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True}]}, {'label': '暂停', 'method': 'animate', 'args': [[None], {'mode': 'immediate'}]}]}], sliders=[{'steps': [{'label': f"{t:.1f}s", 'method': 'animate', 'args': [[f"t={t:.1f}s"], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]} for t in times],'transition': {'duration': 0},'x': 0.1, 'len': 0.9}])
    
    fig.show()

if __name__ == "__main__":
  
    team_strategy_to_visualize = {
        'FY1': {'v': 138.1234, 'theta_deg': -175.5678, 'drops': [(1.1, 4.5), (2.2, 4.8), (3.3, 5.1)]},
        'FY2': {'v': 75.4321, 'theta_deg': 160.9876, 'drops': [(5.5, 3.1), (6.6, 3.4), (7.7, 3.7)]}
    }
    
    visualize_team_strategy(
        team_strategies=team_strategy_to_visualize,
        n_points=1000,
        threshold_percent=70.0
    )