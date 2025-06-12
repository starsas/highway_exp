import numpy as np
from moviepy.editor import ImageSequenceClip
from queue import PriorityQueue  # 用于实现FIFO逻辑
import csv
import gym
import highway_env

def simulate_and_save_videos(env, num_steps=400, env_video_file="./fifo.mp4", output_csv_file="fifo.csv", fps=10):
    # 动作映射表
    action_mapping = {
        "LANE_LEFT": 0,
        "IDLE": 1,
        "LANE_RIGHT": 2,
        "FASTER": 3,
        "SLOWER": 4,
    }

    env_frames = []  # 存储渲染帧
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(num_steps):
            merge_queue = PriorityQueue()  # 优先队列，用于管理车辆合并顺序
            for vehicle in env.road.vehicles:
                print (vehicle.id,vehicle.lane_index)
            # 计算车辆优先级并加入队列
            for vehicle in env.controlled_vehicles:
                if vehicle.lane_index in [("c", "d", 1), ("c", "d", 2)] or vehicle.speed < 0.01:
                    priority_number = 0
                    distance_to_merging_end = env.distance_to_merging_end(vehicle)
                    priority_number -= 2 * (env.ends[2] - distance_to_merging_end) / env.ends[2]
                    priority_number += np.random.rand() * 0.001  # 避免优先级完全相同
                    merge_queue.put((priority_number, vehicle))

            # 初始化所有车辆的默认动作，默认所有车辆保持匀速
            actions = [1 for _ in range(len(env.controlled_vehicles))]

            # 按优先级处理车辆动作
            index = 0
            while not merge_queue.empty():
                vehicle = merge_queue.get()[1]
                if index == 0:
                    actions[vehicle.id] = 3  # 第一辆车加速
                    index += 1
                else:
                    actions[vehicle.id] = 4  # 其他车辆减速

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

            actions = info["new_action"]  # 更新动作，因为由避障算法进行换道
            print("step: ", step, "actions: ", actions)

            if done:  # 检查仿真是否结束
                print(f"Simulation ended at step {step} due to collision or other termination condition.")
                break

    # 将帧序列保存为视频
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    env = gym.make('merge-multi-agent-v0')  # 创建仿真环境
    env.reset()
    simulate_and_save_videos(env, num_steps=400, env_video_file="./fifo.mp4", output_csv_file="fifo.csv", fps=10)
    env.close()