import numpy as np
from moviepy.editor import ImageSequenceClip
import csv
import gym
import highway_env
import os
from queue import PriorityQueue # 导入 PriorityQueue
from custom_llm import LLMAgent, MockVehicle, ACTIONS_ALL, Scenario, is_in_merging_area # 确保导入 is_in_merging_area

def simulate_with_llm_agent(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control.mp4", output_csv_file="llm_merge_control.csv", fps=10):
    env_frames = []
    
    # Action mapping for LLM output to environment input
    action_to_int = {
        'LANE_LEFT': 0,
        'IDLE': 1,
        'LANE_RIGHT': 2,
        'FASTER': 3,
        'SLOWER': 4,
    }

    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed', 'llm_decision', 'llm_target_lane', 'reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(num_steps):
            # 初始化所有车辆的默认动作，默认所有车辆保持匀速
            actions_for_env = [1] * len(env.controlled_vehicles) 
            
            # 创建一个优先级队列来决定CAV的决策顺序
            decision_queue = PriorityQueue()

            # 计算每个受控车辆的优先级并加入队列
            # 优先级策略：最接近合流点的主道车辆优先，其次是匝道车辆。
            # 如果都在合流区内，则离合流点越近优先级越高。
            # 这里实现“最接近合流点的优先”，可以根据需要调整优先级计算方式
            for controlled_veh_obj in env.controlled_vehicles:
                priority_number = 0.0 # 优先级，越小越优先

                # 判断车辆是否在合流区域
                # 需要创建一个临时的MockVehicle来使用is_in_merging_area，因为它期望MockVehicle结构
                temp_mock_vehicle = MockVehicle(controlled_veh_obj) 
                in_merge_area_flag = is_in_merging_area(temp_mock_vehicle)

                # 获取车辆当前在道路上的实际位置（x坐标）
                current_pos_x = controlled_veh_obj.position[0] 
                
                # 合流点的x坐标（假设为env.ends[2]）
                merge_point_x = env.ends[2] 
                
                # 计算到合流点的距离（如果车辆在合流点之前）
                distance_to_merge_point = merge_point_x - current_pos_x

                if controlled_veh_obj.lane_index[2] == 3: # 匝道车 (lane 3)
                    # 匝道车在合流区内更需要决策，且越接近合流点越紧急
                    # 优先级：负的到合流点距离，距离越小（越接近合流点），负数越大，优先级越高
                    priority_number = -distance_to_merge_point 
                    # 额外给匝道车一个小的偏移，确保它在主道车之前被处理（如果距离相同）
                    priority_number -= 1000 # 匝道车优先级更高

                elif controlled_veh_obj.lane_index[2] == 2: # 主道合流车道 (lane 2)
                    # 主道车在合流区内也需要决策
                    priority_number = -distance_to_merge_point
                    # 主道车优先级略低于匝道车，但高于非合流区车辆

                else: # 其他主道车 (lane 0, 1) - 通常优先级较低，或不参与优先级排序
                    # 这些车辆在合流区外，优先级较低
                    priority_number = 10000 + current_pos_x # 越往后优先级越低，保证先处理近合流点的CAV

                # 加入一个小的随机噪声，避免优先级完全相同导致顺序不稳定
                priority_number += np.random.rand() * 0.001 
                
                decision_queue.put((priority_number, controlled_veh_obj))

            # 按优先级从队列中取出车辆并获取LLM决策
            processed_controlled_vehicles_ids = [] # 记录已处理的CAV ID
            while not decision_queue.empty():
                priority, controlled_veh_obj = decision_queue.get()
                
                # 确保每个controlled_veh_obj只被处理一次，以防万一
                if controlled_veh_obj.id in processed_controlled_vehicles_ids:
                    continue
                processed_controlled_vehicles_ids.append(controlled_veh_obj.id)

                # 找到该车辆在env.controlled_vehicles中的索引，以便更新 actions_for_env
                idx = env.controlled_vehicles.index(controlled_veh_obj)

                # 临时将当前受控车辆设置为“ego”
                temp_original_id = getattr(controlled_veh_obj, 'id', f"vehicle_{id(controlled_veh_obj)}")
                controlled_veh_obj.id = "ego" 
                llm_agent.scenario._update_vehicles() # 更新内部mock车辆以反映新的ego赋值

                # 构建观测信息
                current_ego_mock = llm_agent.scenario.vehicles["ego"]
                
                nearby_vehicles = []
                for v_road in env.road.vehicles:
                    if v_road.id != current_ego_mock.id:
                        nearby_v_mock = MockVehicle(v_road)
                        nearby_vehicles.append({
                            'id': nearby_v_mock.id,
                            'speed': nearby_v_mock.speed,
                            'lane_id': nearby_v_mock.lane_id,
                            'lanePosition': nearby_v_mock.lanePosition,
                            'position_xy': v_road.position.tolist(),
                            'is_controlled': nearby_v_mock.is_controlled # 添加车辆类型
                        })
                
                nearby_vehicles.sort(key=lambda x: np.linalg.norm(np.array(x['position_xy']) - controlled_veh_obj.position))

                observation = {
                    'ego_vehicle': {
                        'id': current_ego_mock.id,
                        'speed': current_ego_mock.speed,
                        'lane_id': current_ego_mock.lane_id,
                        'lanePosition': current_ego_mock.lanePosition,
                        'position_xy': controlled_veh_obj.position.tolist(),
                        'in_merging_area': is_in_merging_area(current_ego_mock)
                    },
                    'nearby_vehicles': nearby_vehicles,
                    'road_info': {
                        'ends': env.ends,
                        'num_lanes': len(env.road.network.graph['a']['b'])
                    },
                    'traffic_density': env.config.get("traffic_density", "unknown")
                }
                print(observation['ego_vehicle']['lane_id'],observation['ego_vehicle']['lanePosition'])
                # 从LLM获取决策
                llm_decision_output = llm_agent.get_decision(observation)
                
                llm_action_str = llm_decision_output.get('decision', 'IDLE')
                # target_lane = llm_decision_output.get('target_lane', controlled_veh_obj.lane_index[2] if controlled_veh_obj.lane_index else -1)
                reasoning = llm_decision_output.get('reasoning', 'No specific reasoning provided.')
                
                # 映射LLM的字符串动作到环境的整数动作
                actions_for_env[idx] = action_to_int.get(llm_action_str, 1)

                # 记录数据
                writer.writerow({
                    'step': step,
                    'vehicle_id': temp_original_id,
                    'lane_index': controlled_veh_obj.lane_index[2],
                    'position': controlled_veh_obj.position[0],
                    'speed': controlled_veh_obj.speed,
                    'llm_decision': llm_action_str,
                    # 'llm_target_lane': target_lane,
                    # 'reasoning': reasoning
                })
                print(f"Step: {step}, Controlled Vehicle ID: {temp_original_id}, Decision (LLM): {llm_action_str}")#, Reasoning: {reasoning}
            
                # 恢复车辆ID
                controlled_veh_obj.id = temp_original_id
            
            # 渲染当前环境并保存帧
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)

            # 执行动作
            obs, reward, done, info = env.step(actions_for_env)
            
            print(f"Actual env actions applied by highway_env: {info.get('new_action', 'N/A')}")

            if done:
                print(f"Simulation ended at step {step}")
                break

    # 保存视频
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    # DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_ACTUAL_DEEPSEEK_API_KEY") # 从环境变量获取，或在此处直接设置
    DEEPSEEK_API_KEY = "sk-d181f41df79741bca4b134881c718a9d" # 你的实际API Key

    # if DEEPSEEK_API_KEY == "YOUR_ACTUAL_DEEPSEEK_API_KEY":
    #     print("警告: DEEPSEEK_API_KEY 未设置或使用默认值。请确保您已设置环境变量或在此文件中硬编码您的API Key。")

    env = gym.make('merge-multi-agent-v0')
    env.reset()

    print("受控车辆 (CAVs):", [x.id for x in env.controlled_vehicles])
    print("路上所有车辆:", [x.id for x in env.road.vehicles])

    llm_agent = LLMAgent(api_key=DEEPSEEK_API_KEY, env=env, model_name="deepseek-coder")
    simulate_with_llm_agent(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control.mp4", output_csv_file="llm_merge_control.csv", fps=10)
    env.close()