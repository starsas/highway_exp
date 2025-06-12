import numpy as np
from moviepy.editor import ImageSequenceClip
from queue import PriorityQueue
import csv
import gym
import highway_env

# ============= 安全距离和速度控制 =============
def is_safe_distance(vehicle, other_vehicle, min_distance=10):
    """检查与前方车辆的安全距离"""
    if other_vehicle is None:
        return True
    try:
        distance = other_vehicle.position[0] - vehicle.position[0]
        speed_diff = vehicle.speed - other_vehicle.speed
        # 增加基础安全距离，并加强速度差的影响
        dynamic_distance = min_distance + max(0, speed_diff * 3)  # 增加速度差的影响系数
        return distance >= dynamic_distance
    except (AttributeError, TypeError):
        return True

def should_follow_speed(vehicle, front_vehicle):
    """根据前车速度决定是否跟随"""
    if front_vehicle is None:
        return False
    try:
        return front_vehicle.speed < vehicle.speed * 0.9
    except (AttributeError, TypeError):
        return False

# ============= 车道变换控制 =============
def can_change_lane_left(vehicle, env):
    """检查是否可以向左变道"""
    current_lane = vehicle.lane_index[2]
    if current_lane == 0:  # 已经在最左侧车道
        return False
        
    left_lane = (vehicle.lane_index[0], vehicle.lane_index[1], current_lane - 1)
    v_fl, v_rl = env.road.surrounding_vehicles(vehicle, lane_index=left_lane)
    
    if not is_safe_distance(vehicle, v_fl) or not is_safe_distance(v_rl, vehicle):
        return False
            
    return True

# ============= 速度控制策略 =============
def should_accelerate(vehicle, env):
    """判断是否应该加速"""
    v_fl, _ = env.road.surrounding_vehicles(vehicle)
    if v_fl is None:  # 前方没有车
        return True
    if is_safe_distance(vehicle, v_fl) and v_fl.speed >= vehicle.speed:
        return True
    return False

def should_slow_down(vehicle, env):
    """判断是否应该减速"""
    v_fl, _ = env.road.surrounding_vehicles(vehicle)
    if v_fl is None:
        return False
    return should_follow_speed(vehicle, v_fl)

# ============= 匝道合流控制 =============
def is_merge_vehicle(vehicle):
    """判断是否是匝道车辆"""
    try:
        return vehicle.lane_index[2] == 2
    except (AttributeError, TypeError):
        return False

def can_merge_safely(vehicle, env):
    """判断匝道车辆是否可以安全合流到主车道"""
    target_lane = (vehicle.lane_index[0], vehicle.lane_index[1], 1)
    v_fl, v_rl = env.road.surrounding_vehicles(vehicle, lane_index=target_lane)
    safe_gap = 10
    
    if v_fl and (v_fl.position[0] - vehicle.position[0] < safe_gap or abs(v_fl.speed - vehicle.speed) > 5):
        return False
    if v_rl and (vehicle.position[0] - v_rl.position[0] < safe_gap or abs(v_rl.speed - vehicle.speed) > 5):
        return False
    return True

def check_same_lane_safety(vehicle, env):
    """检查同车道车辆的安全状态"""
    v_fl, v_rl = env.road.surrounding_vehicles(vehicle)
    if v_fl and not is_safe_distance(vehicle, v_fl, min_distance=15):  # 增加同车道安全距离
        return False
    if v_rl and not is_safe_distance(v_rl, vehicle, min_distance=15):
        return False
    return True

def match_speed_to_main(vehicle, env):
    """匝道车辆在安全距离下加速到与主路前车相同速度"""
    target_lane = (vehicle.lane_index[0], vehicle.lane_index[1], 1)
    v_fl, _ = env.road.surrounding_vehicles(vehicle, lane_index=target_lane)
    
    # 首先检查同车道安全
    if not check_same_lane_safety(vehicle, env):
        return 4  # 如果同车道不安全，立即减速
    
    if v_fl:
        distance = v_fl.position[0] - vehicle.position[0]
        speed_diff = vehicle.speed - v_fl.speed
        
        # 增加基础安全距离
        base_safe_distance = 12  # 增加基础安全距离
        dynamic_safe_distance = base_safe_distance + max(0, speed_diff * 2)
        
        if distance < dynamic_safe_distance:
            # 根据距离和速度差决定减速程度
            if distance < base_safe_distance or speed_diff > 3:
                return 4  # 紧急减速
            else:
                return 4  # 轻微减速
        else:
            # 距离安全，根据速度差调整
            if abs(speed_diff) < 1:  # 速度差很小
                return 1  # 保持速度
            elif speed_diff > 0:  # 自身速度大于前车
                if speed_diff > 3:  # 速度差较大
                    return 4  # 减速
                else:
                    return 1  # 保持速度
            else:  # 自身速度小于前车
                if abs(speed_diff) > 3:  # 速度差较大
                    return 3  # 加速
                else:
                    return 1  # 保持速度
                    
    return 1  # 前方无车，保持

# ============= 主车道控制 =============
def should_give_way(vehicle, env):
    """主车道车辆判断是否需要为匝道让行"""
    right_lane = (vehicle.lane_index[0], vehicle.lane_index[1], vehicle.lane_index[2] + 1)
    v_fr, _ = env.road.surrounding_vehicles(vehicle, lane_index=right_lane)
    if v_fr and is_merge_vehicle(v_fr) and can_change_lane_left(vehicle, env):
        return True
    return False

def keep_speed_stable(vehicle, env):
    """主车道车辆保持速度稳定"""
    v_fl, _ = env.road.surrounding_vehicles(vehicle)
    if v_fl:
        distance = v_fl.position[0] - vehicle.position[0]
        speed_diff = vehicle.speed - v_fl.speed
        
        base_safe_distance = 10
        dynamic_safe_distance = base_safe_distance + max(0, speed_diff * 1.5)
        
        if distance < dynamic_safe_distance or speed_diff > 2:
            return 4  # 减速
        elif speed_diff < -2:
            return 3  # 加速
    return 1  # 保持

def is_in_merge_area(vehicle):
    """判断车辆是否在合流区"""
    return vehicle.lane_index in [("c", "d", 1), ("c", "d", 2)]

# ============= 仿真和可视化 =============
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
    
    # 记录上一帧的速度
    last_speeds = {}
    
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed', 'speed_change']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for step in range(num_steps):
            actions = [1 for _ in range(len(env.controlled_vehicles))]
            
            # 处理匝道车辆主动减速逻辑
            for vehicle in env.controlled_vehicles:
                if is_merge_vehicle(vehicle) and is_in_merge_area(vehicle):
                    # 检查同车道安全
                    if not check_same_lane_safety(vehicle, env):
                        actions[vehicle.id] = 4  # 主动减速
                        # 后方同车道车辆也减速
                        for other in env.controlled_vehicles:
                            if (other.lane_index == vehicle.lane_index and
                                other.position[0] < vehicle.position[0] and
                                vehicle.position[0] - other.position[0] < 25):
                                actions[other.id] = 4
            
            # 其余车辆按原有逻辑决策
            for vehicle in env.controlled_vehicles:
                if actions[vehicle.id] != 1:
                    continue  # 已经被上面逻辑赋值
                if is_merge_vehicle(vehicle):
                    if can_merge_safely(vehicle, env) and can_change_lane_left(vehicle, env):
                        actions[vehicle.id] = 0  # 向左变道合流
                    else:
                        actions[vehicle.id] = match_speed_to_main(vehicle, env)
                else:
                    if should_give_way(vehicle, env):
                        actions[vehicle.id] = 0
                    else:
                        actions[vehicle.id] = keep_speed_stable(vehicle, env)
            
            # 记录环境状态
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)
            
            # 执行动作并记录数据
            obs, reward, done, info = env.step(actions)
            
            for vehicle in env.controlled_vehicles:
                current_speed = vehicle.speed
                last_speed = last_speeds.get(vehicle.id, current_speed)
                speed_change = current_speed - last_speed
                last_speeds[vehicle.id] = current_speed
                
                
                writer.writerow({
                    'step': step,
                    'vehicle_id': vehicle.id,
                    'lane_index': vehicle.lane_index[2],
                    'position': vehicle.position[0],
                    'speed': current_speed,
                    'speed_change': speed_change
                })
            
            actions = info["new_action"]
            if done:
                print(f"Simulation ended at step {step} due to collision or other termination condition.")
                break
    
    # 保存视频
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    env = gym.make('merge-multi-agent-v0')
    env.reset()
    simulate_and_save_videos(env, num_steps=400, env_video_file="./fifo1.mp4", output_csv_file="fifo1.csv", fps=10)
    env.close()