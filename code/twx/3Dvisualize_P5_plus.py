import plotly.graph_objects as go
import numpy as np
from numba import njit 

GRAVITY = 9.8
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0

UAV_INITIAL_POS = {
    'FY1': np.array([17800, 0, 1800], dtype=float),
    'FY2': np.array([12000, 1400, 1400], dtype=float),
    'FY3': np.array([6000, -3000, 700], dtype=float),
    'FY4': np.array([11000, 2000, 1800], dtype=float),
    'FY5': np.array([13000, -2000, 1300], dtype=float),
}
MISSILE_INITIAL_POS = np.array([20000, 0, 2000], dtype=float) 
FALSE_TARGET = np.array([0, 0, 0], dtype=float)
TRUE_TARGET_BASE_CENTER = np.array([0, 200, 0], dtype=float)
TRUE_TARGET_RADIUS = 7
TRUE_TARGET_HEIGHT = 10
MISSILE_SPEED = 300.0

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

def visualize_team_strategy(team_strategies, n_points=1000, threshold_percent=70.0):
    uav_names = list(team_strategies.keys())
    num_uavs = len(uav_names)
    total_grenades = sum(len(s['drops']) for s in team_strategies.values())
    print(f"--- 开始生成 {num_uavs} 架无人机, 共 {total_grenades} 枚烟幕弹的协同策略可视化动画 ---")
    print(f"--- 模式: 计算每个无人机的独立贡献时长 ---")

    SIM_DURATION = 80; TIME_STEP = 0.1
    required_occluded_count = int(np.ceil((threshold_percent / 100.0) * n_points))
    target_points = []
    for _ in range(int(n_points * 0.6)):
        theta, z = 2 * np.pi * np.random.rand(), TRUE_TARGET_HEIGHT * np.random.rand()
        target_points.append([TRUE_TARGET_BASE_CENTER[0] + TRUE_TARGET_RADIUS * np.cos(theta), TRUE_TARGET_BASE_CENTER[1] + TRUE_TARGET_RADIUS * np.sin(theta), z])
    for _ in range(int(n_points * 0.4)):
        r, theta = TRUE_TARGET_RADIUS * np.sqrt(np.random.rand()), 2 * np.pi * np.random.rand()
        z = 0 if np.random.rand() < 0.5 else TRUE_TARGET_HEIGHT
        target_points.append([TRUE_TARGET_BASE_CENTER[0] + r * np.cos(theta), TRUE_TARGET_BASE_CENTER[1] + r * np.sin(theta), z])
    target_points = np.array(target_points)

    times = np.arange(0, SIM_DURATION, TIME_STEP)
    trajectory = {}
    missile_velocity_vector = (FALSE_TARGET - MISSILE_INITIAL_POS) / np.linalg.norm(FALSE_TARGET - MISSILE_INITIAL_POS) * MISSILE_SPEED
    trajectory['M1'] = [MISSILE_INITIAL_POS + missile_velocity_vector * t for t in times]
    
    smoke_trajs_by_uav = {name: [] for name in uav_names}
    grenade_trajs_flat = [] 
    
    for uav_name in uav_names:
        strategy = team_strategies[uav_name]
        v = strategy['v']; theta_rad = np.radians(strategy['theta_deg'])
        uav_initial_pos = UAV_INITIAL_POS[uav_name]
        uav_velocity_vector = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
        trajectory[uav_name] = [uav_initial_pos + uav_velocity_vector * t for t in times]
        
        for t_drop, t_delay in strategy['drops']:
            explode_time = t_drop + t_delay
            drop_pos = uav_initial_pos + uav_velocity_vector * t_drop
            explode_pos = drop_pos + uav_velocity_vector * t_delay + np.array([0, 0, -0.5 * GRAVITY * t_delay**2])
            g_traj, s_traj = [], []
            for t in times:
                if t_drop <= t < explode_time:
                    g_traj.append(drop_pos + uav_velocity_vector * (t-t_drop) + np.array([0, 0, -0.5 * GRAVITY * (t-t_drop)**2]))
                else: g_traj.append(np.full(3, np.nan))
                if explode_time <= t < explode_time + SMOKE_DURATION:
                    s_traj.append(explode_pos + np.array([0, 0, -3.0 * (t - explode_time)]))
                else: s_traj.append(np.full(3, np.nan))
            grenade_trajs_flat.append(g_traj)
            smoke_trajs_by_uav[uav_name].append(s_traj)

    cooperative_occlusion_status = []
    cumulative_cooperative_time = []; total_cooperative_time = 0
    individual_occlusion_times = {name: 0.0 for name in uav_names}
    cumulative_individual_times_history = []
    all_occluded_indices_history = []

    for i, t in enumerate(times):
        missile_pos = trajectory['M1'][i]
        total_occluded_indices_this_step = set()
        
        for uav_name in uav_names:
            occluded_indices_for_this_uav = set()
            for smoke_traj in smoke_trajs_by_uav[uav_name]:
                smoke_pos = smoke_traj[i]
                if not np.any(np.isnan(smoke_pos)):
                    for point_idx, tp in enumerate(target_points):
                        if is_line_segment_intersecting_sphere_numba(missile_pos, tp, smoke_pos, SMOKE_RADIUS):
                            occluded_indices_for_this_uav.add(point_idx)
            
            if len(occluded_indices_for_this_uav) >= required_occluded_count:
                individual_occlusion_times[uav_name] += TIME_STEP
            
            total_occluded_indices_this_step.update(occluded_indices_for_this_uav)

        is_coop_occluded = len(total_occluded_indices_this_step) >= required_occluded_count
        cooperative_occlusion_status.append(is_coop_occluded)
        all_occluded_indices_history.append(total_occluded_indices_this_step)
        
        if is_coop_occluded: total_cooperative_time += TIME_STEP
        cumulative_cooperative_time.append(total_cooperative_time)
        cumulative_individual_times_history.append(dict(individual_occlusion_times))

    print(f"\n模拟计算完成。协同策略的总有效遮蔽时长为: {total_cooperative_time:.2f} 秒")
    print("--- 各无人机独立贡献时长 (高精度) ---")
    for uav_name, individual_time in individual_occlusion_times.items():
        print(f"  {uav_name}: {individual_time:.2f} 秒")

    fig = go.Figure()
    uav_colors = {'FY1':'blue', 'FY2':'cyan', 'FY3':'orange', 'FY4':'purple', 'FY5':'magenta'}
    
    fig.add_trace(go.Scatter3d(x=[m[0] for m in trajectory['M1']], y=[m[1] for m in trajectory['M1']], z=[m[2] for m in trajectory['M1']], mode='lines', line=dict(color='red', width=1, dash='dot'), name='M1 弹道'))
    for uav_name in uav_names:
        fig.add_trace(go.Scatter3d(x=[t[0] for t in trajectory[uav_name]], y=[t[1] for t in trajectory[uav_name]], z=[t[2] for t in trajectory[uav_name]], mode='lines', line=dict(color=uav_colors.get(uav_name, 'black'), width=1, dash='dot'), name=f'{uav_name} 轨迹'))
    animated_traces_start_index = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0]], y=[trajectory['M1'][0][1]], z=[trajectory['M1'][0][2]], mode='markers', marker=dict(color='red', size=5), name='M1'))
    for uav_name in uav_names:
        fig.add_trace(go.Scatter3d(x=[trajectory[uav_name][0][0]], y=[trajectory[uav_name][0][1]], z=[trajectory[uav_name][0][2]], mode='markers', marker=dict(color=uav_colors.get(uav_name, 'black'), size=5), name=uav_name))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='magenta', width=1), name='被遮蔽视线'))
    true_target_center_point = TRUE_TARGET_BASE_CENTER + np.array([0, 0, TRUE_TARGET_HEIGHT / 2])
    fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0], true_target_center_point[0]], y=[trajectory['M1'][0][1], true_target_center_point[1]], z=[trajectory['M1'][0][2], true_target_center_point[2]], mode='lines', line=dict(color='lime', width=4), name='主视线 (LOS)'))
    
    for i in range(total_grenades):
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='grey', size=4), name=f'烟幕弹 {i+1}'))
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='green', size=SMOKE_RADIUS, opacity=0.3), name=f'烟幕 {i+1}'))
    
    fig.add_trace(go.Scatter3d(x=[FALSE_TARGET[0]], y=[FALSE_TARGET[1]], z=[FALSE_TARGET[2]], mode='markers', marker=dict(color='black', symbol='x', size=8), name='假目标'))
    u_cyl, h_cyl = np.linspace(0, 2*np.pi, 50), np.linspace(0, TRUE_TARGET_HEIGHT, 2); u, h = np.meshgrid(u_cyl, h_cyl)
    x_cyl, y_cyl, z_cyl = TRUE_TARGET_BASE_CENTER[0] + TRUE_TARGET_RADIUS*np.cos(u), TRUE_TARGET_BASE_CENTER[1] + TRUE_TARGET_RADIUS*np.sin(u), TRUE_TARGET_BASE_CENTER[2] + h
    fig.add_trace(go.Surface(x=x_cyl, y=y_cyl, z=z_cyl, colorscale='Greens', showscale=False, opacity=0.5, name='真目标'))
    fig.add_trace(go.Scatter3d(x=target_points[:, 0], y=target_points[:, 1], z=target_points[:, 2], mode='markers', marker=dict(color='darkgreen', size=1.5), name='目标采样点'))
    
    frames = []
    for i, t in enumerate(times):
        frame_data = [go.Scatter3d(x=[trajectory['M1'][i][0]], y=[trajectory['M1'][i][1]], z=[trajectory['M1'][i][2]])]
        for uav_name in uav_names:
            frame_data.append(go.Scatter3d(x=[trajectory[uav_name][i][0]], y=[trajectory[uav_name][i][1]], z=[trajectory[uav_name][i][2]]))
        
        occluded_los_lines_x, occluded_los_lines_y, occluded_los_lines_z = [], [], []
        if cooperative_occlusion_status[i]:
            for point_idx in list(all_occluded_indices_history[i])[:10]:
                tp = target_points[point_idx]
                occluded_los_lines_x.extend([trajectory['M1'][i][0], tp[0], None]); occluded_los_lines_y.extend([trajectory['M1'][i][1], tp[1], None]); occluded_los_lines_z.extend([trajectory['M1'][i][2], tp[2], None])
        frame_data.append(go.Scatter3d(x=occluded_los_lines_x, y=occluded_los_lines_y, z=occluded_los_lines_z))
        
        los_color = 'magenta' if cooperative_occlusion_status[i] else 'lime'
        status_text = "是" if cooperative_occlusion_status[i] else "否"
        frame_data.append(go.Scatter3d(x=[trajectory['M1'][i][0], true_target_center_point[0]], y=[trajectory['M1'][i][1], true_target_center_point[1]], z=[trajectory['M1'][i][2], true_target_center_point[2]], line=dict(color=los_color)))

        flat_smoke_trajs = [item for sublist in smoke_trajs_by_uav.values() for item in sublist]
        for j in range(total_grenades):
            frame_data.append(go.Scatter3d(x=[grenade_trajs_flat[j][i][0]], y=[grenade_trajs_flat[j][i][1]], z=[grenade_trajs_flat[j][i][2]]))
            frame_data.append(go.Scatter3d(x=[flat_smoke_trajs[j][i][0]], y=[flat_smoke_trajs[j][i][1]], z=[flat_smoke_trajs[j][i][2]]))

        individual_times_str = " | ".join([f"{name}: {cumulative_individual_times_history[i][name]:.2f}s" for name in uav_names])
        title = (f"战场模拟 | 时间: {t:.1f}s | 协同遮蔽({threshold_percent}%): <b style='color:{los_color};'>{status_text}</b>"
                 f"<br>累计(协同): {cumulative_cooperative_time[i]:.2f}s | 累计(独立): {individual_times_str}")
        
        frame = go.Frame(data=frame_data, name=f"t={t:.1f}s", traces=list(range(animated_traces_start_index, len(fig.data)-3)), layout={'title': title})
        frames.append(frame)
    fig.frames = frames

    initial_individual_str = " | ".join([f"{name}: 0.00s" for name in uav_names])
    initial_title = f"战场模拟 | 时间: 0.0s | 协同遮蔽({threshold_percent}%): 否<br>累计(协同): 0.00s | 累计(独立): {initial_individual_str}"
    
    fig.update_layout(title=initial_title, scene=dict(xaxis=dict(title='X轴 (m)', range=[-1000, 21000]), yaxis=dict(title='Y轴 (m)', range=[-4000, 4000]), zaxis=dict(title='Z轴 (m)', range=[0, 2200]), aspectmode='manual', aspectratio=dict(x=5, y=2, z=0.8)), updatemenus=[{'type': 'buttons','buttons': [{'label': '播放', 'method': 'animate', 'args': [None, {'frame': {'duration': 1000*TIME_STEP, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True}]}, {'label': '暂停', 'method': 'animate', 'args': [[None], {'mode': 'immediate'}]}]}], sliders=[{'steps': [{'label': f"{t:.1f}s", 'method': 'animate', 'args': [[f"t={t:.1f}s"], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]} for t in times],'transition': {'duration': 0},'x': 0.1, 'len': 0.9}])
    
    fig.show()

if __name__ == "__main__":

    team_strategy_to_visualize = {
        'FY1': {'v': 74.9, 'theta_deg': 177.7, 'drops': [(0.1, 2.5), (1.5, 2.6)]},
        'FY3': {'v': 139.6, 'theta_deg': 122.5, 'drops': [(18.9, 7.6)]},
        'FY4': {'v': 100.0, 'theta_deg': -150.0, 'drops': [(10.0, 5.0), (12.0, 5.0), (14.0, 5.0)]}
    }
    
    visualize_team_strategy(
        team_strategies=team_strategy_to_visualize,
        n_points=1000,
        threshold_percent=70.0
    )