import numpy as np
from moviepy.editor import ImageSequenceClip
import csv
import gym
import highway_env
from highway_env.envs.common.mdp_controller import mdp_controller
from highway_env.utils import rotated_rectangles_intersect
import copy

def predict_vehicle_trajectory(vehicle, action, env_copy, steps=5):
    """预测车辆未来轨迹"""
    vehicle_copy = copy.deepcopy(vehicle)
    trajectory = []
    
    for _ in range(steps):
        mdp_controller(vehicle_copy, env_copy, action)
        trajectory.append([vehicle_copy.position, vehicle_copy.heading, vehicle_copy.speed])
    
    return trajectory

def check_collision(vehicle, other, other_trajectory):
    """检查是否发生碰撞"""
    if vehicle.crashed or other is vehicle:
        return False
        
    # 欧氏距离快速检查
    if np.linalg.norm(other_trajectory[0] - vehicle.position) > vehicle.LENGTH:
        return False
        
    # 精确的矩形碰撞检查
    return rotated_rectangles_intersect(
        (vehicle.position, 0.9 * vehicle.LENGTH, 0.9 * vehicle.WIDTH, vehicle.heading),
        (other_trajectory[0], 0.9 * other.LENGTH, 0.9 * other.WIDTH, other_trajectory[1])
    )

def check_safety(vehicle, action, env_copy):
    """检查动作是否安全"""
    # 预测车辆轨迹
    vehicle_trajectory = predict_vehicle_trajectory(vehicle, action, env_copy)
    
    # 获取周围车辆
    v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle)
    v_fr, v_rr = None, None
    
    # 获取侧向车道信息
    side_lanes = env_copy.road.network.side_lanes(vehicle.lane_index)
    if side_lanes:
        v_fr, v_rr = env_copy.road.surrounding_vehicles(vehicle, side_lanes[0])
    
    # 检查每个时间步的安全性
    for t in range(len(vehicle_trajectory)):
        vehicle_state = vehicle_trajectory[t]
        
        # 检查前方车辆
        if v_fl:
            v_fl_trajectory = predict_vehicle_trajectory(v_fl, v_fl.action if hasattr(v_fl, 'action') else 1, env_copy)
            if t < len(v_fl_trajectory):
                if check_collision(vehicle, v_fl, v_fl_trajectory[t]):
                    return False
                if v_fl_trajectory[t][0][0] - vehicle_state[0][0] < 10:
                    return False
        
        # 检查后方车辆
        if v_rl:
            v_rl_trajectory = predict_vehicle_trajectory(v_rl, v_rl.action if hasattr(v_rl, 'action') else 1, env_copy)
            if t < len(v_rl_trajectory):
                if check_collision(vehicle, v_rl, v_rl_trajectory[t]):
                    return False
                if vehicle_state[0][0] - v_rl_trajectory[t][0][0] < 10:
                    return False
        
        # 检查变道安全
        if action in [0, 2]:  # 变道动作
            target_lane = None
            if action == 0 and vehicle.lane_index[2] > 0:  # 向左变道
                target_lane = (vehicle.lane_index[0], vehicle.lane_index[1], vehicle.lane_index[2] - 1)
            elif action == 2 and vehicle.lane_index[2] < 2:  # 向右变道
                target_lane = (vehicle.lane_index[0], vehicle.lane_index[1], vehicle.lane_index[2] + 1)
            
            if target_lane:
                v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle, target_lane)
                
                if v_fl:
                    v_fl_trajectory = predict_vehicle_trajectory(v_fl, v_fl.action if hasattr(v_fl, 'action') else 1, env_copy)
                    if t < len(v_fl_trajectory):
                        if check_collision(vehicle, v_fl, v_fl_trajectory[t]):
                            return False
                        if v_fl_trajectory[t][0][0] - vehicle_state[0][0] < 15:
                            return False
                
                if v_rl:
                    v_rl_trajectory = predict_vehicle_trajectory(v_rl, v_rl.action if hasattr(v_rl, 'action') else 1, env_copy)
                    if t < len(v_rl_trajectory):
                        if check_collision(vehicle, v_rl, v_rl_trajectory[t]):
                            return False
                        if vehicle_state[0][0] - v_rl_trajectory[t][0][0] < 15:
                            return False
    
    return True

def get_safe_action(vehicle, env, env_copy):
    """获取安全动作"""
    # 获取可用动作
    available_actions = []
    
    # 检查变道可能性
    if vehicle.lane_index[2] > 0:  # 可以向左变道
        available_actions.append(0)
    if vehicle.lane_index[2] < 2:  # 可以向右变道
        available_actions.append(2)
    
    # 检查速度调整可能性
    if vehicle.speed < vehicle.SPEED_MAX:
        available_actions.append(3)
    if vehicle.speed > 0:
        available_actions.append(4)
    
    # IDLE动作总是可用
    available_actions.append(1)
    
    # 检查每个动作的安全性
    safe_actions = []
    for action in available_actions:
        if check_safety(vehicle, action, env_copy):
            safe_actions.append(action)
    
    # 如果没有安全动作，返回IDLE
    return safe_actions if safe_actions else [1]

def simulate_and_save_videos(env, num_steps=400, env_video_file="./fifo1.mp4", output_csv_file="fifo1.csv", fps=10):
    """运行仿真并保存视频和CSV数据"""
    action_mapping = {
        "LANE_LEFT": 0,
        "IDLE": 1,
        "LANE_RIGHT": 2,
        "FASTER": 3,
        "SLOWER": 4,
    }
    env_frames = []
    
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for step in range(num_steps):
            # 初始化所有车辆的默认动作
            actions = [1 for _ in range(len(env.controlled_vehicles))]
            
            # 创建环境副本用于预测
            env_copy = copy.deepcopy(env)
            
            # 处理每个车辆的动作
            for vehicle in env.controlled_vehicles:
                # 获取安全动作列表
                safe_actions = get_safe_action(vehicle, env, env_copy)
                
                # 根据车辆类型选择动作
                if vehicle.lane_index[2] == 2:  # 匝道车辆
                    if 0 in safe_actions:  # 如果可以安全变道
                        actions[vehicle.id] = 0
                    else:
                        actions[vehicle.id] = 4  # 减速
                else:  # 主路车辆
                    if 3 in safe_actions:  # 如果可以安全加速
                        actions[vehicle.id] = 3
                    else:
                        actions[vehicle.id] = 1  # 保持速度
            
            # 渲染当前环境并保存帧
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)
            
            # 执行动作并记录车辆状态
            obs, reward, done, info = env.step(actions)
            for vehicle in env.controlled_vehicles:
                writer.writerow({
                    'step': step,
                    'vehicle_id': vehicle.id,
                    'lane_index': vehicle.lane_index[2],
                    'position': vehicle.position[0],
                    'speed': vehicle.speed
                })
            
            if done:
                print(f"Simulation ended at step {step} due to collision or other termination condition.")
                break
    
    # 将帧序列保存为视频
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    env = gym.make('merge-multi-agent-v0')
    env.reset()
    simulate_and_save_videos(env, num_steps=400, env_video_file="./fifo1.mp4", output_csv_file="fifo1.csv", fps=10)
    env.close()