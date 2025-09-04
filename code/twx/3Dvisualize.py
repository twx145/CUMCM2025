import plotly.graph_objects as go
import numpy as np

# --- 1. 模拟参数和决策变量 (基于问题1的设定) ---

# -- 场景设置 --
SIM_DURATION = 65
TIME_STEP = 0.1  # 减小步长以提高计时精度
GRAVITY = 9.8

# -- 决策变量 --
UAV_NAME = 'FY1'
UAV_SPEED = 120.0
uav_initial_pos = np.array([17800, 0, 1800])
target_pos = np.array([0, 0, 0])
direction_vector = target_pos - uav_initial_pos
direction_vector[2] = 0
uav_velocity_vector = (direction_vector / np.linalg.norm(direction_vector)) * UAV_SPEED

# -- 干扰弹策略 --
DROP_TIME = 1.5
EXPLODE_DELAY = 3.6
EXPLODE_TIME = DROP_TIME + EXPLODE_DELAY
SMOKE_DURATION = 20.0
SMOKE_RADIUS = 10.0

# -- 目标和威胁数据 --
missiles = {'M1': np.array([20000, 0, 2000])}
uavs = {'FY1': uav_initial_pos}
false_target = np.array([0, 0, 0])
missile_speed = 300.0

true_target_base_center = np.array([0, 200, 0])
true_target_radius = 7
true_target_height = 10
# 为简化计算，我们将真目标视线点设为其中心
true_target_los_point = true_target_base_center + np.array([0, 0, true_target_height / 2])


# --- 2. 遮蔽判断核心函数 ---
def is_occluded(missile_pos, smoke_center_pos, target_pos, smoke_radius):
    """判断从导弹到目标的视线是否被烟幕球体遮挡"""
    if np.any(np.isnan(smoke_center_pos)):
        return False
    
    # 使用向量法计算点到线段的距离
    line_vec = target_pos - missile_pos
    point_vec = smoke_center_pos - missile_pos
    
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0: # 导弹和目标在同一点
        return np.linalg.norm(point_vec) <= smoke_radius

    t = np.dot(point_vec, line_vec) / line_len_sq
    
    if t < 0.0: # 投影点在线段起点之外，最近点是起点
        closest_point = missile_pos
    elif t > 1.0: # 投影点在线段终点之外，最近点是终点
        closest_point = target_pos
    else: # 投影点在线段上
        closest_point = missile_pos + t * line_vec
        
    distance = np.linalg.norm(smoke_center_pos - closest_point)
    
    return distance <= smoke_radius

# --- 3. 轨迹和状态预计算 ---
times = np.arange(0, SIM_DURATION, TIME_STEP)
trajectory = {}
occlusion_status = []
cumulative_occlusion_time = []
total_occluded_time = 0

# (与之前代码相同的轨迹计算...)
m1_initial_pos = missiles['M1']
m1_velocity_vector = (false_target - m1_initial_pos) / np.linalg.norm(false_target - m1_initial_pos) * missile_speed
trajectory['M1'] = [m1_initial_pos + m1_velocity_vector * t for t in times]
trajectory['FY1'] = [uav_initial_pos + uav_velocity_vector * t for t in times]
drop_pos = uav_initial_pos + uav_velocity_vector * DROP_TIME
drop_velocity = uav_velocity_vector
grenade_traj = []
smoke_cloud_traj = []
explode_pos = drop_pos + drop_velocity * (EXPLODE_TIME - DROP_TIME) + np.array([0, 0, -0.5 * GRAVITY * (EXPLODE_TIME - DROP_TIME)**2])
for t in times:
    if t >= DROP_TIME and t < EXPLODE_TIME:
        dt = t - DROP_TIME
        grenade_traj.append(drop_pos + drop_velocity * dt + np.array([0, 0, -0.5 * GRAVITY * dt**2]))
    else:
        grenade_traj.append(np.full(3, np.nan))
    if t >= EXPLODE_TIME and t < EXPLODE_TIME + SMOKE_DURATION:
        dt_sink = t - EXPLODE_TIME
        smoke_cloud_traj.append(explode_pos + np.array([0, 0, -3.0 * dt_sink]))
    else:
        smoke_cloud_traj.append(np.full(3, np.nan))
trajectory['Grenade'] = grenade_traj
trajectory['SmokeCloud'] = smoke_cloud_traj

# 逐帧计算遮蔽状态和累计时间
for i, t in enumerate(times):
    status = is_occluded(trajectory['M1'][i], trajectory['SmokeCloud'][i], true_target_los_point, SMOKE_RADIUS)
    occlusion_status.append(status)
    if status:
        total_occluded_time += TIME_STEP
    cumulative_occlusion_time.append(total_occluded_time)

print(f"模拟计算完成。最终有效遮蔽总时长为: {total_occluded_time:.2f} 秒")

# --- 4. 创建 Plotly 动画 ---

fig = go.Figure()

# -- 添加初始帧的物体 --
# (实体)
fig.add_trace(go.Scatter3d(x=[u[0] for u in trajectory['FY1']], y=[u[1] for u in trajectory['FY1']], z=[u[2] for u in trajectory['FY1']], mode='lines', line=dict(color='blue', width=1, dash='dot'), name='FY1 轨迹'))
fig.add_trace(go.Scatter3d(x=[m[0] for m in trajectory['M1']], y=[m[1] for m in trajectory['M1']], z=[m[2] for m in trajectory['M1']], mode='lines', line=dict(color='red', width=1, dash='dot'), name='M1 弹道'))
# (动态点)
fig.add_trace(go.Scatter3d(x=[trajectory['M1'][0][0]], y=[trajectory['M1'][0][1]], z=[trajectory['M1'][0][2]], mode='markers', marker=dict(color='red', size=5), name='M1'))
fig.add_trace(go.Scatter3d(x=[trajectory['FY1'][0][0]], y=[trajectory['FY1'][0][1]], z=[trajectory['FY1'][0][2]], mode='markers', marker=dict(color='blue', size=5), name='FY1'))
fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='grey', size=4), name='烟幕弹'))
fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode='markers', marker=dict(color='green', size=SMOKE_RADIUS, opacity=0.3), name='烟幕云团'))
# (视线)
los_x = [trajectory['M1'][0][0], true_target_los_point[0]]
los_y = [trajectory['M1'][0][1], true_target_los_point[1]]
los_z = [trajectory['M1'][0][2], true_target_los_point[2]]
fig.add_trace(go.Scatter3d(x=los_x, y=los_y, z=los_z, mode='lines', line=dict(color='lime', width=4), name='视线 (LOS)'))


# -- 添加静态物体 --
# (假目标)
fig.add_trace(go.Scatter3d(x=[false_target[0]], y=[false_target[1]], z=[false_target[2]], mode='markers', marker=dict(color='black', symbol='x', size=8), name='假目标'))
# (真目标圆柱体)
u_cyl = np.linspace(0, 2 * np.pi, 50)
h_cyl = np.linspace(0, true_target_height, 2)
u, h = np.meshgrid(u_cyl, h_cyl)
x_cyl = true_target_base_center[0] + true_target_radius * np.cos(u)
y_cyl = true_target_base_center[1] + true_target_radius * np.sin(u)
z_cyl = true_target_base_center[2] + h
fig.add_trace(go.Surface(x=x_cyl, y=y_cyl, z=z_cyl, colorscale='Greens', showscale=False, opacity=0.5, name='真目标'))

# -- 创建动画帧 --
frames = []
for i, t in enumerate(times):
    # 动态点的位置
    frame_data = [
        go.Scatter3d(x=[trajectory['M1'][i][0]], y=[trajectory['M1'][i][1]], z=[trajectory['M1'][i][2]]),
        go.Scatter3d(x=[trajectory['FY1'][i][0]], y=[trajectory['FY1'][i][1]], z=[trajectory['FY1'][i][2]]),
        go.Scatter3d(x=[trajectory['Grenade'][i][0]], y=[trajectory['Grenade'][i][1]], z=[trajectory['Grenade'][i][2]]),
        go.Scatter3d(x=[trajectory['SmokeCloud'][i][0]], y=[trajectory['SmokeCloud'][i][1]], z=[trajectory['SmokeCloud'][i][2]]),
    ]
    # 视线 (LOS) 的位置和颜色
    los_x = [trajectory['M1'][i][0], true_target_los_point[0]]
    los_y = [trajectory['M1'][i][1], true_target_los_point[1]]
    los_z = [trajectory['M1'][i][2], true_target_los_point[2]]
    los_color = 'magenta' if occlusion_status[i] else 'lime'
    frame_data.append(go.Scatter3d(x=los_x, y=los_y, z=los_z, mode='lines', line=dict(color=los_color, width=4)))

    # 动态更新的标题
    status_text = "是" if occlusion_status[i] else "否"
    frame_layout = {
        'title': f"战场模拟 | 时间: {t:.1f}s | 是否遮蔽: <b style='color:{los_color};'>{status_text}</b> | 累计遮蔽: {cumulative_occlusion_time[i]:.2f}s"
    }
    
    frame = go.Frame(
        data=frame_data,
        name=f"t={t:.1f}s",
        # 这里的traces指定了要更新fig.data中的哪几个元素
        traces=[2, 3, 4, 5, 6],
        layout=frame_layout
    )
    frames.append(frame)

fig.frames = frames

# --- 5. 设置动画播放器和布局 ---
# (与之前代码相同)
def create_animation_settings():
    return {
        'transition': {'duration': 0},
        'frame': {'duration': 1000 * TIME_STEP, 'redraw': True},
        'fromcurrent': True
    }

fig.update_layout(
    title=f"战场模拟 | 时间: 0.0s | 是否遮蔽: 否 | 累计遮蔽: 0.00s",
    scene=dict(
        xaxis=dict(title='X轴 (m)', range=[-1000, 21000]),
        yaxis=dict(title='Y轴 (m)', range=[-4000, 4000]),
        zaxis=dict(title='Z轴 (m)', range=[0, 2200]),
        aspectmode='manual',
        aspectratio=dict(x=5, y=2, z=0.8)
    ),
    updatemenus=[{'type': 'buttons','buttons': [
            {'label': '播放', 'method': 'animate', 'args': [None, create_animation_settings()]},
            {'label': '暂停', 'method': 'animate', 'args': [[None], {'mode': 'immediate'}]}
        ]}],
    sliders=[{'steps': [
            {'label': f"{t:.1f}s", 'method': 'animate', 'args': [[f"t={t:.1f}s"], create_animation_settings()]} for t in times
        ],'transition': {'duration': 0},'x': 0.1, 'len': 0.9}]
)

fig.show()