# import numpy as np
# from moviepy.editor import ImageSequenceClip
# import csv
# import gym
# import highway_env

# class VirtualQueue:
#     def __init__(self):
#         self.vehicles = []  # 存储合流区的车辆
#         self.positions = []  # 存储车辆位置
#         self.speeds = []    # 存储车辆速度
#         self.virtual_positions = []  # 存储虚拟队列中的位置
#         self.safe_distance = 10  # 安全距离
#         self.merge_safe_distance = 15  # 合流安全距离
#         self.formation_spacing = 12  # 编队间距
#         self.k_p = 0.5  # 位置控制增益
#         self.k_v = 0.3  # 速度控制增益
        
#     def project_to_virtual_queue(self, vehicle, env):
#         """将车辆投影到虚拟队列中"""
#         if vehicle.lane_index[2] == 1:  # 主道车
#             return vehicle.position[0]
#         else:  # 匝道车
#             merge_point = env.ends[2]  # 合流点位置
#             current_pos = vehicle.position[0]
#             distance_to_merge = merge_point - current_pos
#             return merge_point - distance_to_merge
            
#     def update(self, env):
#         """更新虚拟队列，记录合流区的车辆"""
#         self.vehicles = []
#         self.positions = []
#         self.speeds = []
#         self.virtual_positions = []
        
#         # 收集合流区的车辆信息
#         for vehicle in env.controlled_vehicles:
#             if env.distance_to_merging_end(vehicle) < env.ends[2] * 0.3:  # 在合流区
#                 self.vehicles.append(vehicle)
#                 self.positions.append(vehicle.position[0])
#                 self.speeds.append(vehicle.speed)
#                 self.virtual_positions.append(self.project_to_virtual_queue(vehicle, env))
                
#         # 根据虚拟位置排序
#         sorted_indices = np.argsort(self.virtual_positions)
#         self.vehicles = [self.vehicles[i] for i in sorted_indices]
#         self.positions = [self.positions[i] for i in sorted_indices]
#         self.speeds = [self.speeds[i] for i in sorted_indices]
#         self.virtual_positions = [self.virtual_positions[i] for i in sorted_indices]
        
#     def check_lane_change_safety(self, vehicle, target_lane, env):
#         """检查换道安全性"""
#         front_vehicle = None
#         rear_vehicle = None
#         min_front_distance = float('inf')
#         min_rear_distance = float('inf')
        
#         for v in env.controlled_vehicles:
#             if v.lane_index[2] == target_lane:
#                 distance = v.position[0] - vehicle.position[0]
#                 if distance > 0 and distance < min_front_distance:  # 前车
#                     front_vehicle = v
#                     min_front_distance = distance
#                 elif distance < 0 and abs(distance) < min_rear_distance:  # 后车
#                     rear_vehicle = v
#                     min_rear_distance = abs(distance)
        
#         if front_vehicle and min_front_distance < self.merge_safe_distance:
#             return False
#         if rear_vehicle and min_rear_distance < self.merge_safe_distance:
#             return False
            
#         return True
        
#     def get_formation_control(self, vehicle, env):
#         """编队控制"""
#         try:
#             idx = self.vehicles.index(vehicle)
#         except ValueError:
#             return vehicle.speed
            
#         # 获取前后车信息
#         front_vehicle = self.vehicles[idx-1] if idx > 0 else None
#         rear_vehicle = self.vehicles[idx+1] if idx < len(self.vehicles)-1 else None
        
#         # 计算与前后车的虚拟距离
#         front_virtual_distance = float('inf')
#         if front_vehicle:
#             front_virtual_distance = abs(self.virtual_positions[idx] - self.virtual_positions[idx-1])
            
#         rear_virtual_distance = float('inf')
#         if rear_vehicle:
#             rear_virtual_distance = abs(self.virtual_positions[idx] - self.virtual_positions[idx+1])
            
#         # 计算期望速度
#         target_speed = vehicle.speed
        
#         # 编队控制
#         if front_vehicle:
#             spacing_error = front_virtual_distance - self.formation_spacing
#             speed_error = front_vehicle.speed - vehicle.speed
#             target_speed += self.k_p * spacing_error + self.k_v * speed_error
            
#         return target_speed
        
#     def get_vehicle_action(self, vehicle, env):
#         """根据车辆类型和位置决定动作"""
#         # 1. 检查是否在合流区
#         in_merge_area = env.distance_to_merging_end(vehicle) < env.ends[2] * 0.3
        
#         # 2. 获取前后车信息
#         front_vehicle = None
#         rear_vehicle = None
#         front_distance = float('inf')
#         rear_distance = float('inf')
        
#         for v in env.controlled_vehicles:
#             if v.lane_index[2] == vehicle.lane_index[2]:  # 同车道
#                 distance = v.position[0] - vehicle.position[0]
#                 if distance > 0 and distance < front_distance:  # 前车
#                     front_vehicle = v
#                     front_distance = distance
#                 elif distance < 0 and abs(distance) < rear_distance:  # 后车
#                     rear_vehicle = v
#                     rear_distance = abs(distance)
        
#         # 3. 决策逻辑
#         action = 1  # 默认IDLE
        
#         if in_merge_area:  # 在合流区
#             if vehicle.lane_index[2] == 1:  # 主道车
#                 # 检查右侧是否有匝道车
#                 has_merge_vehicle = False
#                 merge_vehicle_distance = float('inf')
                
#                 for v in env.controlled_vehicles:
#                     if v.lane_index[2] == 2:  # 匝道车
#                         distance = v.position[0] - vehicle.position[0]
#                         if abs(distance) < self.merge_safe_distance:
#                             has_merge_vehicle = True
#                             merge_vehicle_distance = distance
#                             break
                
#                 if has_merge_vehicle:
#                     # 检查左侧是否有空位
#                     if self.check_lane_change_safety(vehicle, 0, env):
#                         action = 0  # LANE_LEFT
#                     else:
#                         # 减速等待匝道车变道
#                         action = 4  # SLOWER
                        
#             elif vehicle.lane_index[2] == 2:  # 匝道车
#                 # 检查是否可以换到主道
#                 if self.check_lane_change_safety(vehicle, 1, env):
#                     action = 2  # LANE_RIGHT
#                 else:
#                     # 检查主道是否有车阻挡
#                     has_blocking_vehicle = False
#                     for v in env.controlled_vehicles:
#                         if v.lane_index[2] == 1:  # 主道车
#                             distance = v.position[0] - vehicle.position[0]
#                             if abs(distance) < self.merge_safe_distance:
#                                 has_blocking_vehicle = True
#                                 break
                    
#                     if has_blocking_vehicle:
#                         action = 4  # SLOWER
#                     else:
#                         # 使用编队控制
#                         target_speed = self.get_formation_control(vehicle, env)
#                         if target_speed > vehicle.speed:
#                             action = 3  # FASTER
#                         elif target_speed < vehicle.speed:
#                             action = 4  # SLOWER
#                         else:
#                             action = 1  # IDLE
                            
#         else:  # 不在合流区
#             # 保持安全距离
#             if front_distance < self.safe_distance:
#                 action = 4  # SLOWER
#             elif rear_distance < self.safe_distance:
#                 action = 3  # FASTER
#             else:
#                 action = 1  # IDLE
                
#         return action

# def simulate_and_save_videos(env, num_steps=400, env_video_file="./merge_control.mp4", output_csv_file="merge_control.csv", fps=10):
#     # 动作映射表
#     action_mapping = {
#         "LANE_LEFT": 0,
#         "IDLE": 1,
#         "LANE_RIGHT": 2,
#         "FASTER": 3,
#         "SLOWER": 4,
#     }

#     # 创建虚拟队列
#     virtual_queue = VirtualQueue()
    
#     env_frames = []
#     with open(output_csv_file, mode='w', newline='') as csvfile:
#         fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'virtual_position', 'speed', 'target_speed']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         for step in range(num_steps):
#             # 更新虚拟队列
#             virtual_queue.update(env)
            
#             # 初始化所有车辆的默认动作
#             actions = [1 for _ in range(len(env.controlled_vehicles))]
            
#             # 为每个车辆计算动作
#             for vehicle in env.controlled_vehicles:
#                 # 获取车辆动作
#                 actions[vehicle.id] = virtual_queue.get_vehicle_action(vehicle, env)
                
#                 # 记录数据
#                 writer.writerow({
#                     'step': step,
#                     'vehicle_id': vehicle.id,
#                     'lane_index': vehicle.lane_index[2],
#                     'position': vehicle.position[0],
#                     'virtual_position': virtual_queue.virtual_positions[virtual_queue.vehicles.index(vehicle)] if vehicle in virtual_queue.vehicles else 0,
#                     'speed': vehicle.speed,
#                     'target_speed': virtual_queue.get_formation_control(vehicle, env) if vehicle in virtual_queue.vehicles else vehicle.speed
#                 })

#             # 渲染当前环境并保存帧
#             env_frame = env.render(mode='rgb_array')
#             env_frames.append(env_frame)

#             # 执行动作
#             obs, reward, done, info = env.step(actions)
#             actions = info["new_action"]
#             print(f"Step: {step}, Actions: {actions}")

#             if done:
#                 print(f"Simulation ended at step {step}")
#                 break

#     # 保存视频
#     env_clip = ImageSequenceClip(env_frames, fps=fps)
#     env_clip.write_videofile(env_video_file, codec="libx264")
#     print(f"Environment video saved as {env_video_file}")

# if __name__ == "__main__":
#     env = gym.make('merge-multi-agent-v0')
#     env.reset()
#     simulate_and_save_videos(env, num_steps=400, env_video_file="./merge_control.mp4", output_csv_file="merge_control.csv", fps=10)
#     env.close()

import numpy as np
from moviepy.editor import ImageSequenceClip
from queue import PriorityQueue
import csv
import gym
import highway_env

# 全局安全参数
SAFE_DISTANCE = 15  # 安全跟车距离

def is_in_merging_area(vehicle):
    """判断车辆是否在合流区域"""
    return vehicle.lane_index in [("c", "d", 1), ("c", "d", 2)]

def get_surrounding_vehicles(env, vehicle, lane):
    """获取目标车道前后车辆信息"""
    return env.road.neighbour_vehicles(vehicle, lane)

def can_change_lane(env, vehicle, direction):
    """判断是否可以安全变道"""
    current_lane = vehicle.lane_index[2]
    target_lane = current_lane + (-1 if direction == "left" else 1)
    
    # 检查目标车道是否存在
    if target_lane < 0 or target_lane >= 3:
        print(f"车辆{vehicle.id}: 不存在目标车道")
        return False
    
    # 获取目标车道前后车辆
    front_vehicle, rear_vehicle = get_surrounding_vehicles(env, vehicle, 
        (vehicle.lane_index[0], vehicle.lane_index[1], target_lane))
    
    # 计算安全间距
    safe_front = True
    safe_rear = True

    if front_vehicle:
        front_distance = front_vehicle.position[0] - vehicle.position[0]
        safe_front = front_distance > SAFE_DISTANCE*1.7

    if rear_vehicle:
        rear_distance = vehicle.position[0] - rear_vehicle.position[0]
        safe_rear = rear_distance > SAFE_DISTANCE*1.5


    return safe_front and safe_rear

def maintain_safe_distance(env, vehicle):
    """保持安全跟车距离"""
    front_vehicle, rear_vehicle = get_surrounding_vehicles(env, vehicle, vehicle.lane_index)
    
    # 检查前车
    if front_vehicle is None:
        return 3  # FASTER (加速)
    else:
        front_distance = front_vehicle.position[0] - vehicle.position[0]
        if front_distance < SAFE_DISTANCE*1.2:
            return 4  # SLOWER
        elif vehicle.lane_index[2] == 0 and front_distance > SAFE_DISTANCE * 1.5:
            return 3  # FASTER (加速)
        elif vehicle.lane_index[2] == 1 and front_distance > SAFE_DISTANCE * 1.5:
            return 3  
    
    # 检查后车
    if rear_vehicle:
        rear_distance = vehicle.position[0] - rear_vehicle.position[0]
        if rear_distance < SAFE_DISTANCE:
            return 4  # SLOWER (减速)
    
    return 1  # IDLE (匀速)

def simulate_and_save_videos(env, num_steps=400, 
                           env_video_file="./fifo2.mp4", 
                           output_csv_file="fifo2.csv", 
                           fps=10):
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
            merge_queue = PriorityQueue()
            ramp_vehicles = []
            main_road_vehicles = []
            temp_queue = []  # 临时列表用于打印
            p=[]
            print("control_veh:",[x.id for x in env.controlled_vehicles])
            print("all_veh:",[x.id for x in env.road.vehicles])
            # 车辆分类处理
            for vehicle in env.road.vehicles:
                if vehicle.lane_index[2] == 0 or vehicle.lane_index[2] == 1:  # 匝道车道
                    main_road_vehicles.append(vehicle)
                else:
                    ramp_vehicles.append(vehicle)

            # 计算优先级并记录车辆信息
            for vehicle in env.controlled_vehicles:
                if is_in_merging_area(vehicle):
                    print("has in merge area")
                    # 计算优先级
                    priority_number = 0
                    distance_to_merging_end = env.distance_to_merging_end(vehicle)
                    priority_number -= 2 * (env.ends[2] - distance_to_merging_end) / env.ends[2]
                    priority_number += np.random.rand() * 0.001  # 避免优先级完全相同
                    
                    # 获取合流区内最近的前后车信息
                    front_distance = float('inf')
                    rear_distance = float('inf')
                    front_type = None
                    rear_type = None
                    
                    # 遍历所有合流区车辆
                    for v in env.controlled_vehicles:
                        if v.id != vehicle.id and is_in_merging_area(v):
                            distance = v.position[0] - vehicle.position[0]
                            if distance > 0 and distance < front_distance:  # 前车
                                front_distance = distance
                                front_type = "main" if v in main_road_vehicles else "ramp"
                            elif distance < 0 and abs(distance) < rear_distance:  # 后车
                                rear_distance = abs(distance)
                                rear_type = "main" if v in main_road_vehicles else "ramp"
                   
                    # 存储车辆信息
                    vehicle_info = {
                        "type": "main" if vehicle in main_road_vehicles else "ramp",
                        "front_distance": front_distance,
                        "rear_distance": rear_distance,
                        "front_type": front_type,
                        "rear_type": rear_type
                    }
                    print(vehicle_info)
                    merge_queue.put((priority_number, vehicle, vehicle_info))
                    temp_queue.append((priority_number, vehicle.id))  # 记录优先级和车辆ID
                    
            
            # 打印优先级队列中的车辆顺序
            temp_queue.sort()  # 按优先级排序
            # print(f"Step {step} 优先级队列车辆顺序（从低到高）: ", [f"车辆{vid}" for _, vid in temp_queue])

            # 初始化默认动作
            actions = [action_mapping["IDLE"] for _ in range(len(env.controlled_vehicles))]

            # 主道非合流区车辆控制

            for i, vehicle in enumerate(main_road_vehicles):
                if not is_in_merging_area(vehicle):
                    if vehicle in env.controlled_vehicles:
                        # 后续车辆保持安全距离
                        actions[vehicle.id] = maintain_safe_distance(env, vehicle)
            # 合流区车辆处理
            index = 0
            while not merge_queue.empty():
                _, vehicle, info = merge_queue.get()
                # 1. 队列头车加速
                if index == 0:
                    actions[vehicle.id] = action_mapping["FASTER"]
                    index += 1
                    continue
                print(f"车辆{vehicle.id} info: {info['type']}")
                # 2. 根据车辆类型设置动作
                if info["type"] == "main":  # 主道车
                    # 检查前后匝道车
                    has_merge_vehicle_front = False
                    has_merge_vehicle_rear = False
                    
                    if info["front_type"] == "ramp":
                        if info["front_distance"] < SAFE_DISTANCE:
                            has_merge_vehicle_front = True
                    
                    if info["rear_type"] == "ramp":
                        if info["rear_distance"] < SAFE_DISTANCE:
                            has_merge_vehicle_rear = True
                    
                    if has_merge_vehicle_front or has_merge_vehicle_rear:
                        # 检查左侧是否有空位
                        if can_change_lane(env, vehicle, "left"):
                            actions[vehicle.id] = action_mapping["LANE_LEFT"]
                        else:
                            # 根据与匝道车的相对位置决定动作
                            if has_merge_vehicle_front:  # 匝道车在前
                                actions[vehicle.id] = action_mapping["SLOWER"]
                            elif has_merge_vehicle_rear:  # 匝道车在后
                                actions[vehicle.id] = action_mapping["FASTER"]
                            else:
                                actions[vehicle.id] = action_mapping["IDLE"]
                    else:
                        # 检查前后车距离和类型
                        if info["front_distance"] < SAFE_DISTANCE:
                            actions[vehicle.id] = action_mapping["SLOWER"]
                        elif info["rear_distance"] < SAFE_DISTANCE:
                            actions[vehicle.id] = action_mapping["IDLE"]
                        else:
                            actions[vehicle.id] = action_mapping["IDLE"]
                            
                elif info["type"] == "ramp":  # 匝道车
                    print(f"车辆{vehicle.id}: 匝道车,info: {info['front_distance']}, {info['rear_distance']}")
                    # 优先考虑变道
                    if can_change_lane(env, vehicle, "left"):
                        print(f"车辆{vehicle.id}: 变道")
                        actions[vehicle.id] = action_mapping["LANE_LEFT"]
                    else:
                        # 检查前方车辆类型
                        if info["front_type"] == "main":  # 前方是主道车
                            actions[vehicle.id] = action_mapping["SLOWER"]
                        else:  # 前方是匝道车
                            if info["front_distance"] < SAFE_DISTANCE:
                                actions[vehicle.id] = action_mapping["SLOWER"]
                            else:
                                actions[vehicle.id] = action_mapping["IDLE"]
                else:
                    actions[vehicle.id] = action_mapping["IDLE"]
                # # 3. 级联减速处理
                # if actions[vehicle.id] == action_mapping["SLOWER"]:
                #     for v in env.controlled_vehicles:
                #         if v.lane_index[2] == vehicle.lane_index[2] and \
                #            v.position[0] > vehicle.position[0] and \
                #            v.position[0] - vehicle.position[0] < SAFE_DISTANCE * 2:
                #             actions[v.id] = action_mapping["SLOWER"]

            # 环境交互
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)
            
            obs, reward, done, infos = env.step(actions)
            for vehicle in env.controlled_vehicles:
                writer.writerow({
                    'step': step,
                    'vehicle_id': vehicle.id,
                    'lane_index': vehicle.lane_index[2],
                    'position': vehicle.position[0],
                    'speed': vehicle.speed
                })
            actions = infos["new_action"]  # 更新动作，因为由避障算法进行换道
            print("step: ", step, "actions: ", actions)
            if done:
                print(f"Simulation ended at step {step}")
                break

    # 视频生成
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Video saved as {env_video_file}")

if __name__ == "__main__":
    env = gym.make('merge-multi-agent-v0')
    env.reset()
    simulate_and_save_videos(env)
    env.close()
