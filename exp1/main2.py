import numpy as np
from moviepy.editor import ImageSequenceClip
import csv
import gym
import highway_env
import os
from queue import PriorityQueue 

# Make sure your custom_llm.py is correctly named and importable
from predict_llm import LLMAgent, MockVehicle, ACTIONS_ALL, Scenario, is_in_merging_area 
import re 

def simulate_and_save_videos(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control.mp4", output_csv_file="llm_merge_control.csv", fps=10):
    """
    Simulates the highway environment with an LLM agent controlling CAVs,
    and saves the simulation video and vehicle data.

    :param env: The highway_env environment instance.
    :param llm_agent: The LLMAgent instance for decision making.
    :param num_steps: Total simulation steps.
    :param env_video_file: Output path for the simulation video.
    :param output_csv_file: Output path for vehicle data CSV.
    :param fps: Frames per second for the output video.
    """
    env_frames = []
    
    # 将LLM的字符串动作映射到环境的整数动作
    action_to_int = {
        'LANE_LEFT': 0,
        'IDLE': 1,
        'LANE_RIGHT': 2,
        'FASTER': 3,
        'SLOWER': 4,
    }

    with open(output_csv_file, mode='w', newline='') as csvfile:
        # CSV 字段列表，只包含 LLM 最终决策的 'decision' 和 'reasoning'
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed', 'llm_decision', 'reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(num_steps):
            # current_step_cav_decisions 字典用于存储当前仿真步中，优先级靠前的CAV已经做出的决策。
            # 这些决策会被注入到 MockVehicle 的 'decision' 属性中，供后续CAV的LLM推理使用。
            current_step_cav_decisions: Dict[str, str] = {}
            
            # --- 构建优先级队列 ---
            decision_queue = PriorityQueue()
            for vehicle in env.road.vehicles:
                print (vehicle.id,vehicle.lane_index)

            for vehicle in env.controlled_vehicles: # 遍历所有受控车辆
                priority_number = 0.0 
                
                # 创建一个临时的 MockVehicle 来使用 is_in_merging_area，因为它期望 MockVehicle 结构。
                # 注意：这里 vehicle 是 highway_env 的原始 Vehicle 对象，MockVehicle 是包装器。
                temp_mock_vehicle_for_priority_check = MockVehicle(vehicle) 

                current_pos_x = vehicle.position[0] 
                
                # 获取合流点的x坐标（根据 merge_env_v1.py 的定义，ends[2]是合流结束点）
                merge_point_x = env.ends[2] 
                
                # 计算车辆当前位置到合流点的距离
                distance_to_merge_point = merge_point_x - current_pos_x

                # 优先级策略：
                # 越接近合流点（x值越大），且在合流区内的车，优先级越高。
                # 匝道车通常比主道车更需要优先决策。
                if vehicle.lane_index[2] == 3: # 匝道车 (lane 3)
                    # 匝道车在合流区内更需要决策，且越接近合流点越紧急
                    # 优先级：负的到合流点距离，距离越小（越接近合流点），负数越大，优先级越高
                    priority_number = -distance_to_merge_point 
                    # 额外给匝道车一个大的负值偏移，确保它在主道车之前被处理（即使距离稍远）
                    priority_number -= 1000 # 确保匝道车优先级最高

                elif vehicle.lane_index[2] == 2: # 主道合流车道 (lane 2)
                    # 主道车在合流区内也需要决策，但优先级略低于匝道车
                    priority_number = -distance_to_merge_point 

                else: # 其他主道车 (lane 0, 1) - 通常优先级较低，或不参与紧急决策
                    # 这些车辆在合流区外，优先级最低，根据其位置排序
                    priority_number = 10000 + current_pos_x # 越往后（X越大）优先级越低，但都在合流车之后

                # 加入一个小的随机噪声，避免优先级完全相同导致顺序不稳定
                priority_number += np.random.rand() * 0.001 
                
                decision_queue.put((priority_number, vehicle))
            # --- 优先级队列构建结束 ---

            # 初始化所有受控车辆的动作，默认为 IDLE
            actions_for_env = [1] * len(env.controlled_vehicles) 
            
            processed_controlled_vehicles_ids = [] 
            while not decision_queue.empty():
                priority, controlled_veh_obj = decision_queue.get()
                
                if controlled_veh_obj.id in processed_controlled_vehicles_ids:
                    continue
                processed_controlled_vehicles_ids.append(controlled_veh_obj.id)

                idx_in_controlled_vehicles = env.controlled_vehicles.index(controlled_veh_obj)

                # 临时将当前受控车辆的ID设置为 "ego"
                temp_original_id = getattr(controlled_veh_obj, 'id', f"vehicle_{id(controlled_veh_obj)}")
                controlled_veh_obj.id = "ego" 
                
                # ！！！在调用 LLM Agent 之前，更新 Scenario 中的 MockVehicle 状态 ！！！
                # 这会将 `current_step_cav_decisions` 中的决策注入到对应的 MockVehicle 中。
                # LLM 的工具（如 CheckTrajectoryConflict）将能够访问这些最新的决策。
                llm_agent.scenario._update_vehicles(current_step_cav_decisions) 
                
                # 构建 LLM 观察信息 (只包含 ego 自身和道路基本信息，周围车辆信息由LLM通过工具获取)
                current_ego_mock = llm_agent.scenario.vehicles["ego"] # 从更新后的 scenario 中获取当前 ego 的 MockVehicle

                observation_for_llm = {
                    'ego_vehicle': {
                        'id': current_ego_mock.id,
                        'speed': current_ego_mock.speed,
                        'lane_id': current_ego_mock.lane_idx, 
                        'lane_id_tuple': current_ego_mock.lane_id_tuple, 
                        'lanePosition': current_ego_mock.lanePosition,
                        'position_xy': controlled_veh_obj.position.tolist(), # Real vehicle position
                        'in_merging_area': is_in_merging_area(current_ego_mock)
                    },
                    'road_info': {
                        'ends': env.ends, 
                        'num_lanes': 4
                    },
                    'traffic_density': env.config.get("traffic_density", "unknown")
                }

                # 调用 LLM Agent 获取决策
                llm_decision_output = llm_agent.get_decision(observation_for_llm)
                
                llm_action_str = llm_decision_output.get('decision', 'IDLE')
                reasoning = llm_decision_output.get('reasoning', 'No specific reasoning provided.')
                
                # 将 LLM 的字符串动作映射到环境的整数动作
                actions_for_env[idx_in_controlled_vehicles] = action_to_int.get(llm_action_str, 1) 
                
                # ！！！及时更新当前 CAV 的决策，供后续优先级靠后的 CAV 使用 ！！！
                current_step_cav_decisions[temp_original_id] = llm_action_str

                # 记录数据
                writer.writerow({
                    'step': step,
                    'vehicle_id': temp_original_id,
                    'lane_index': controlled_veh_obj.lane_index[2],
                    'position': controlled_veh_obj.position[0],
                    'speed': controlled_veh_obj.speed,
                    'llm_decision': llm_action_str,
                    'reasoning': reasoning
                })
                print(f"Step: {step}, Controlled Vehicle ID: {temp_original_id}, Decision (LLM): {llm_action_str}, Reasoning: {reasoning}")
            
                # 恢复车辆的原始ID，以便环境的正常运行
                controlled_veh_obj.id = temp_original_id
            
            # 渲染当前环境并保存帧
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)

            # 执行所有受控车辆的动作
            obs, reward, done, info = env.step(actions_for_env)
            
            print(f"Actual env actions applied by highway_env: {info.get('new_action', 'N/A')}")

            if done:
                print(f"Simulation ended at step {step}")
                break

    # 将帧序列保存为视频
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    DEEPSEEK_API_KEY = "sk-d181f41df79741bca4b134881c718a9d"

    # 创建仿真环境
    env = gym.make('merge-multi-agent-v0')
    env.reset()

    # 打印初始的受控车辆和所有车辆ID，用于调试
    print("受控车辆 (CAVs):", [x.id for x in env.controlled_vehicles])
    print("路上所有车辆:", [x.id for x in env.road.vehicles])

    # 初始化 LLM Agent
    llm_agent = LLMAgent(api_key=DEEPSEEK_API_KEY, env=env, model_name="deepseek-coder")

    # 运行仿真
    simulate_and_save_videos(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control1.mp4", output_csv_file="llm_merge_control.csv", fps=10)
    env.close()