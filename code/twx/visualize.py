import plotly.graph_objects as go
import numpy as np

# --- 1. 定义题目给出的所有初始位置数据 ---

# 目标位置
false_target = {'name': '假目标 (原点)', 'pos': (0, 0, 0)}
true_target_base_center = (0, 200, 0)
true_target_radius = 7
true_target_height = 10

# 导弹初始位置
missiles = {
    'M1': (20000, 0, 2000),
    'M2': (19000, 600, 2100),
    'M3': (18000, -600, 1900)
}

# 无人机初始位置
uavs = {
    'FY1': (17800, 0, 1800),
    'FY2': (12000, 1400, 1400),
    'FY3': (6000, -3000, 700),
    'FY4': (11000, 2000, 1800),
    'FY5': (13000, -2000, 1300)
}

# --- 2. 创建 Plotly 3D 图形 ---

fig = go.Figure()

# --- 3. 绘制无人机 (UAVs) ---
uav_positions = list(uavs.values())
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in uav_positions],
    y=[p[1] for p in uav_positions],
    z=[p[2] for p in uav_positions],
    mode='markers+text',
    text=list(uavs.keys()),
    textposition='top center',
    marker=dict(
        size=8,
        color='blue',  # 无人机为蓝色
        symbol='circle',
        opacity=0.8
    ),
    name='我方无人机 (UAVs)'
))

# --- 4. 绘制导弹 (Missiles) ---
missile_positions = list(missiles.values())
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in missile_positions],
    y=[p[1] for p in missile_positions],
    z=[p[2] for p in missile_positions],
    mode='markers+text',
    text=list(missiles.keys()),
    textposition='top center',
    marker=dict(
        size=8,
        color='red',  # 导弹为红色
        symbol='diamond',
        opacity=0.8
    ),
    name='来袭导弹 (Missiles)'
))

# --- 5. 绘制假目标 (原点) ---
fig.add_trace(go.Scatter3d(
    x=[false_target['pos'][0]],
    y=[false_target['pos'][1]],
    z=[false_target['pos'][2]],
    mode='markers',
    marker=dict(
        size=10,
        color='black',
        symbol='cross'
    ),
    name=false_target['name']
))

# --- 6. 绘制真目标 (圆柱体) ---
# 为了可视化圆柱体，我们生成其表面的点
u = np.linspace(0, 2 * np.pi, 50) # 角度
h = np.linspace(0, true_target_height, 20) # 高度
u, h = np.meshgrid(u, h)

# 圆柱体表面坐标
x_cyl = true_target_base_center[0] + true_target_radius * np.cos(u)
y_cyl = true_target_base_center[1] + true_target_radius * np.sin(u)
z_cyl = true_target_base_center[2] + h

fig.add_trace(go.Surface(
    x=x_cyl, y=y_cyl, z=z_cyl,
    colorscale='Greens',
    showscale=False,
    opacity=0.5,
    name='真目标'
))


# --- 7. 绘制导弹的飞行轨迹 (直线) ---
for name, pos in missiles.items():
    fig.add_trace(go.Scatter3d(
        x=[pos[0], false_target['pos'][0]],
        y=[pos[1], false_target['pos'][1]],
        z=[pos[2], false_target['pos'][2]],
        mode='lines',
        line=dict(
            color='red',
            width=2,
            dash='dash'
        ),
        name=f'弹道 {name}'
    ))

# --- 8. 更新图形布局和样式 ---
fig.update_layout(
    title='战场初始态势三维可视化',
    scene=dict(
        xaxis_title='X 轴 (m)',
        yaxis_title='Y 轴 (m)',
        zaxis_title='Z 轴 (m)',
        # 'data' 模式会根据数据范围自动调整轴比例，最真实地反映空间关系
        # 但可能因为X轴范围远大于Y/Z轴而导致图形被压扁，可以用'auto'或'cube'获得更好的视觉效果
        aspectmode='data' 
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend_title="图例"
)

# --- 9. 显示图形 ---
fig.show()