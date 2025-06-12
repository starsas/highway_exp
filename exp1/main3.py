import numpy as np
from moviepy.editor import ImageSequenceClip
import csv
import gym
import highway_env
import os
from queue import PriorityQueue 

# Make sure your custom_llm.py is correctly named and importable
from two_llm import LLMAgent, MockVehicle, ACTIONS_ALL, Scenario, is_in_merging_area 
import re 

# --- Constants for Lane/Road Structure (Reflect the new 2 main + 1 ramp model) ---
# These should ideally be defined in a shared config or passed from env
# For this main.py, we mirror the custom_llm.py's understanding
NUM_TOTAL_LANES = 3 # Total lanes: 0 (main), 1 (main), 2 (ramp)
MAIN_LANE_INDICES = [0, 1]
RAMP_LANE_IDX = 2 # The ramp is now explicitly lane 2
MERGE_MAIN_LANE_IDX = 1 # Main lane involved in merge is lane 1
MERGE_RAMP_LANE_IDX = 2 # Ramp lane involved in merge is lane 2


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
        llm_agent.scenario._update_vehicles() 

        for step in range(num_steps):
            current_step_cav_decisions: Dict[str, str] = {}
            
            decision_queue = PriorityQueue()
            actions_for_env = [1] * len(env.controlled_vehicles) # Initialize all controlled vehicles to IDLE

            for vehicle in env.controlled_vehicles:
                priority_number = 0.0 
                temp_mock_vehicle_for_priority_check = MockVehicle(vehicle) 

                current_pos_x = vehicle.position[0] 
                merge_point_x = env.ends[2] 
                distance_to_merge_point = merge_point_x - current_pos_x

                if vehicle.lane_index[2] == RAMP_LANE_IDX: # 匝道车 (lane 2)
                    priority_number = -distance_to_merge_point 
                    priority_number -= 2000 
                elif vehicle.lane_index[2] == MERGE_MAIN_LANE_IDX: # 主道合并车道 (lane 1)
                    priority_number = -distance_to_merge_point 
                    priority_number -= 1000 
                else: # 其他主道车 (lane 0)
                    priority_number = 10000 + current_pos_x 

                priority_number += np.random.rand() * 0.001 
                
                decision_queue.put((priority_number, vehicle))

            processed_controlled_vehicles_ids = [] 
            while not decision_queue.empty():
                priority, controlled_veh_obj = decision_queue.get()
                
                if controlled_veh_obj.id in processed_controlled_vehicles_ids:
                    continue
                processed_controlled_vehicles_ids.append(controlled_veh_obj.id)

                idx_in_controlled_vehicles = env.controlled_vehicles.index(controlled_veh_obj)
                
                v_rear,v_front=env.road.neighbour_vehicles(controlled_veh_obj)
                if controlled_veh_obj.lane_index[2]!=2 and v_front==None:
                    #主道头车加速
                    llm_action_str=3
                    reasoning="Vechicle in the front, FASTER"

                elif controlled_veh_obj.lane_index[0]=="a":
                    #头段IDLE
                    llm_action_str=1
                    reasoning="Vechicle in lane'a''b',keep IDLE"
                    
                else:
                    
                    #在ab段IDLE
                    
                    # Store original ID before modification
                    original_controlled_veh_id_before_ego_set = getattr(controlled_veh_obj, 'id', f"vehicle_{id(controlled_veh_obj)}")
                    
                    # --- CRITICAL STEP: Temporarily set ID to "ego" ---
                    controlled_veh_obj.id = "ego" 
                    
                    # --- UPDATE SCENARIO TO REFLECT TEMPORARY "EGO" ID ---
                    llm_agent.scenario._update_vehicles(current_step_cav_decisions) 

                    # --- DEBUGGING LOGS ---
                    print(f"  Processing CAV: Original ID '{original_controlled_veh_id_before_ego_set}', Now temporarily 'ego'")
                    if "ego" not in llm_agent.scenario.vehicles:
                        print(f"    ERROR: 'ego' key NOT found in llm_agent.scenario.vehicles AFTER _update_vehicles!")
                        # This is the root cause if it prints!
                        print(f"    Scenario vehicles keys: {list(llm_agent.scenario.vehicles.keys())}")
                        # Fallback directly
                        llm_decision_output = llm_agent._fallback_decision(
                            {'ego_vehicle': {'lane_id': controlled_veh_obj.lane_index[2]}}, # Provide minimal info for fallback
                            reason=f"KeyError: 'ego' in scenario.vehicles for original ID {original_controlled_veh_id_before_ego_set}"
                        )
                    else:
                        current_ego_mock = llm_agent.scenario.vehicles["ego"]

                        # Build observation for LLM
                        observation_for_llm = {
                            'ego_vehicle': {
                                'id': current_ego_mock.id,
                                'speed': current_ego_mock.speed,
                                'lane_id': current_ego_mock.lane_idx, 
                                'lane_id_tuple': current_ego_mock.lane_id_tuple, 
                                'lanePosition': current_ego_mock.lanePosition,
                                'position_xy': controlled_veh_obj.position.tolist(),
                                'in_merging_area': is_in_merging_area(current_ego_mock)
                            },
                            'road_info': {
                                'ends': env.ends, 
                                'num_lanes': NUM_TOTAL_LANES # Use updated total lane count
                            },
                            'traffic_density': env.config.get("traffic_density", "unknown")
                        }

                        llm_decision_output = llm_agent.get_decision(observation_for_llm)
                    
                    llm_action_str = llm_decision_output.get('decision', 'IDLE')
                    reasoning = llm_decision_output.get('reasoning', 'No specific reasoning provided.')
                    
                    actions_for_env[idx_in_controlled_vehicles] = action_to_int.get(llm_action_str, 1) 
                    
                    current_step_cav_decisions[original_controlled_veh_id_before_ego_set] = llm_action_str # Use original ID here
               
                writer.writerow({
                    'step': step,
                    'vehicle_id': idx_in_controlled_vehicles,
                    'lane_index': controlled_veh_obj.lane_index[2],
                    'position': controlled_veh_obj.position[0],
                    'speed': controlled_veh_obj.speed,
                    'llm_decision': llm_action_str,
                    'reasoning': reasoning
                })
                print(f"Step: {step}, Controlled Vehicle ID: {idx_in_controlled_vehicles}, Pos: {controlled_veh_obj.lane_index}, Decision (LLM): {llm_action_str}")
            
                # --- CRITICAL STEP: Restore original ID ---
                controlled_veh_obj.id = idx_in_controlled_vehicles
            
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)

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
    # 从环境变量获取 DeepSeek API Key，如果没有设置则使用默认值
    DEEPSEEK_API_KEY = "sk-d181f41df79741bca4b134881c718a9d"
    # 创建仿真环境
    env = gym.make('merge-multi-agent-v0')
    env.reset()

    print("受控车辆 (CAVs):", [x.id for x in env.controlled_vehicles])
    print("路上所有车辆:", [x.id for x in env.road.vehicles])

    # 初始化 LLM Agent
    llm_agent = LLMAgent(api_key=DEEPSEEK_API_KEY, env=env, model_name="deepseek-coder")

    # 运行仿真
    simulate_and_save_videos(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control2.mp4", output_csv_file="llm_merge_control2.csv", fps=10)
    env.close()