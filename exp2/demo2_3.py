import numpy as np
import gym
import highway_env
from highway_env.vehicle.controller import MDPVehicle
from highway_env.envs.common.mdp_controller import mdp_controller
from moviepy.editor import ImageSequenceClip
from highway_env.utils import rotated_rectangles_intersect
import  highway_env.utils as utils
import copy
import csv

def predict(vehicle, action, env):
    env_copy = copy.deepcopy(env)
    vehicle_copy = copy.deepcopy(vehicle)
    for _ in range(2):  # 模拟短时间内的状态变化
        mdp_controller(vehicle_copy, env_copy, action)
    return {
        'position': vehicle_copy.position.copy(),
        'speed': vehicle_copy.speed,
        'lane_index': vehicle_copy.lane_index,
        'heading': vehicle_copy.heading
    }
def check_collision(vehicle, other):
    # 预测其他车辆的状态
    other_predicted = predict(other, 1, env)
    
    # 如果车辆已经发生碰撞，或者是同一辆车，则返回 False
    if vehicle.crashed or other is vehicle:
        return False
    
    # 计算两车之间的距离
    distance = np.linalg.norm(np.array(other_predicted['position']) - np.array(vehicle.position))
    
    # 如果距离大于车辆长度，则不发生碰撞
    if distance > (vehicle.LENGTH + other.LENGTH) / 2:  # 使用车辆长度的一半作为碰撞检测的阈值
        return False
    
    # 使用旋转矩形相交检测
    return rotated_rectangles_intersect(
        (vehicle.position, vehicle.LENGTH, vehicle.WIDTH, vehicle.heading),
        (other_predicted['position'], other.LENGTH, other.WIDTH, other_predicted['heading'])
    )

def is_safe_state(predicted_state, vehicle, other_vehicles):
    for other in other_vehicles:
        distance = np.linalg.norm(np.array(predicted_state['position']) - np.array(other.position))
        relative_speed = abs(predicted_state['speed'] - other.speed)
        safe_distance = max(5.0, relative_speed * 0.5)
        if distance < safe_distance:
            return False
        if predicted_state['lane_index'] == other.lane_index and distance < 15.0:
            return False
    return 0 < predicted_state['speed'] <= 30

def cal_reward(vehicle,action, other_vehicles, env):
    min_distance = float('inf')
    # for other in other_vehicles:
    #     distance = np.linalg.norm(np.array(vehicle.position) - np.array(other.position))
    #     min_distance = min(min_distance, distance)
    # if min_distance < 5.0:
    #     crashed = 1
    # else:
    #     crashed = 0
    scaled_speed = utils.lmap(vehicle.speed, env.config["reward_speed_range"], [0, 1])
    crashed = 0
    count = 0
    for other in other_vehicles:
        print("check:——vehicle",vehicle.id,"other",other.id)
        count += 1
        crashed = 1 if check_collision(vehicle, other) else 0
        if crashed == 1:
            print("vehicle",vehicle.id,"collision with vehicle",other.id)
    print("count",count)
        #如果车辆位于匝道上，则计算该车道的代价，代价为基于车辆与匝道尽头距离的指数函数
    if vehicle.lane_index == ("c", "d", 2):
        Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(env.ends[:3])) ** 2 / (
                10 * env.ends[2]))
    else:
        Merging_lane_cost = 0

        # lane change cost to avoid unnecessary/frequent lane changes
    Lane_change_cost = -1*10 * env.config["LANE_CHANGE_COST"] if action == 0 or action == 2 else 0

        #计算车辆的车头时距，并根据车头时距计算一个代价，车头时距除以 某个时间常量与车速的乘积，如果车速为0，则代价为0
    headway_distance = env._compute_headway_distance(vehicle)
    Headway_cost = np.log(
        headway_distance / (env.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        ##计算总的代价，由碰撞代价+高速代价+匝道代价+车头时距代价四部分组成
    reward = env.config["COLLISION_REWARD"] * (-1 * crashed) \
                 + (-1 * env.config["HIGH_SPEED_REWARD"] * np.clip(1 - scaled_speed, 0, 1)) \
                 + env.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + env.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0) \
                    + Lane_change_cost
    return reward

def cal_predicted_reward(vehicle, action, predicted_state, other_vehicles, env):
    if not is_safe_state(predicted_state, vehicle, other_vehicles):
        return float('-inf')

    original_state = {
        'position': vehicle.position.copy(),
        'speed': vehicle.speed,
        'lane_index': vehicle.lane_index
    }

    vehicle.position = predicted_state['position']
    vehicle.speed = predicted_state['speed']
    vehicle.lane_index = predicted_state['lane_index']

    reward = cal_reward(vehicle, action, other_vehicles, env)

    vehicle.position = original_state['position']
    vehicle.speed = original_state['speed']
    vehicle.lane_index = original_state['lane_index']
    return reward

def get_surroundings(vehicle, env, max_distance=10.0):
    others = []
    v_fl, v_rl = env.road.surrounding_vehicles(vehicle)
    if isinstance(v_fl, MDPVehicle): others.append(v_fl)
    if isinstance(v_rl, MDPVehicle): others.append(v_rl)
    for lane in env.road.network.side_lanes(vehicle.lane_index):
        v_fr, v_rr = env.road.surrounding_vehicles(vehicle, lane_index=lane)
        if isinstance(v_fr, MDPVehicle): others.append(v_fr)
        if isinstance(v_rr, MDPVehicle): others.append(v_rr)

    # 过滤出在 max_distance 范围内的车辆
    others = [v for v in others if np.linalg.norm(np.array(vehicle.position) - np.array(v.position)) <= max_distance]
    return others

def cal_best_reward_nash(vehicle1, vehicle2,env):
     best_reward = float('-inf')
     best_action = 1
     for a in range(5):
         s1 = predict(vehicle1, a, env)
         u1 = cal_predicted_reward(vehicle1, a, s1, [vehicle2], env)
         if u1 > best_reward:
            best_reward = u1
            best_action = a
     return best_reward,best_action
def nash_bargaining(vehicle1, vehicle2, env):
    d1,b1 = cal_best_reward_nash(vehicle1, vehicle2, env)
    d2,b2 = cal_best_reward_nash(vehicle2, vehicle1, env)
    best_product = float('-inf')
    best_actions = (b1, b2)
    print(best_actions)
    for a1 in range(5):
        for a2 in range(5):
            s1 = predict(vehicle1, a1, env)
            s2 = predict(vehicle2, a2, env)
            u1 = cal_predicted_reward(vehicle1, a1, s1, [vehicle2], env)
            u2 = cal_predicted_reward(vehicle2, a2, s2, [vehicle1], env)
            if u1 == float('-inf') or u2 == float('-inf'):
                continue
            product = (u1 - d1) * (u2 - d2)
            if product > best_product:
                best_product = product
                best_actions = (a1, a2)
    print(best_actions)
    return best_actions

def simulation(env, num_steps=400, video_path="./nash_result2.mp4", csv_path="./nash_data.csv", fps=10):
    action_mapping = {
        "LANE_LEFT": 0,
        "IDLE": 1,
        "LANE_RIGHT": 2,
        "FASTER": 3,
        "SLOWER": 4,
    }
    frames = []
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["step", "vehicle_id", "lane", "pos_x", "pos_y", "speed", "reward"])
        writer.writeheader()
        cooperative = False
        for step in range(num_steps):
            actions = [1] * len(env.controlled_vehicles)
            processed_vehicles = [0] * len(env.controlled_vehicles)
            for i, vehicle in enumerate(env.controlled_vehicles):
                if processed_vehicles[i] == 0:
                   
                    others = get_surroundings(vehicle, env, max_distance=10.0)  # 设置距离限制
                    for other in others:
                        if cooperative:
                            a1, a2 = nash_bargaining(vehicle, other, env)
                        else:
                            _,a1= cal_best_reward_nash(vehicle, other, env)
                        actions[i] = a1
                        if processed_vehicles[other.id] == 1:
                            continue
                        if cooperative:
                            actions[other.id] = a2
                            processed_vehicles[other.id] = 1
                        else:
                            actions[other.id] = cal_best_reward_nash(other, vehicle, env)[1]
                            processed_vehicles[other.id] = 1
                    # if others:
                    #     closest = min(others, key=lambda v: np.linalg.norm(np.array(v.position) - np.array(vehicle.position)))
                    #     if cooperative:
                    #         a1, a2 = nash_bargaining(vehicle, closest, env)
                    #     else:
                    #         _, a1 = cal_best_reward_nash(vehicle, closest, env)
                    #     actions[i] = a1
                    #     if closest in env.controlled_vehicles:
                    #         idx = env.controlled_vehicles.index(closest)
                    #         if processed_vehicles[idx] == 1:
                    #             continue
                    #         if cooperative:
                    #             actions[idx] = a2
                    #         else:
                    #             actions[idx] = cal_best_reward_nash(closest, vehicle, env)[1]
                    #         processed_vehicles[idx] = 1

            obs, reward, done, info = env.step(actions)
            frames.append(env.render(mode="rgb_array"))
            print("step: ", step, "actions: ", info["new_action"])
            print(info['agents_rewards'])
            for i, vehicle in enumerate(env.controlled_vehicles):
                writer.writerow({
                    "step": step,
                    "vehicle_id": i,
                    "lane": vehicle.lane_index,
                    "pos_x": vehicle.position[0],
                    "pos_y": vehicle.position[1],
                    "speed": vehicle.speed,
                    "reward": cal_reward(vehicle, actions[i], get_surroundings(vehicle, env), env)
                })
            if done:
                break

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(video_path, codec="libx264")

# 用法示例
if __name__ == "__main__":
    env = gym.make('merge-multi-agent-v0')
    # env.configure({
    #     "vehicles_count": 10,
    #     "duration": 60,
    #     "simulation_frequency": 15,
    #     "policy_frequency": 5,
    #     "COLLISION_REWARD": -10,
    #     "MERGING_LANE_COST": 2.0,
    #     "HIGH_SPEED_REWARD": 0.5
    # })
    env.reset()
    simulation(env)
