import numpy as np
from moviepy.editor import ImageSequenceClip
import csv
import gym
import highway_env
import os
from queue import PriorityQueue 
from typing import Any, Dict, List, Tuple, Optional
import datetime

# Make sure your custom_llm.py is correctly named and importable
from two_llm import LLMAgent, MockVehicle, ACTIONS_ALL, Scenario, is_in_merging_area 
# Import constants from custom_llm for consistency and clarity
from two_llm import NUM_MAIN_LANES, MAIN_LANE_INDICES, RAMP_LANE_IDX, MERGE_MAIN_LANE_IDX, NUM_TOTAL_LANES,MERGE_RAMP_LANE_IDX 
import re 

def simulate_and_save_videos(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control.mp4", output_csv_file="llm_merge_control.csv", fps=10, log_file_path="./llm_reasoning_log.txt"):
    """
    Simulates the highway environment with an LLM agent controlling CAVs,
    and saves the simulation video, vehicle data, and LLM reasoning log.

    :param env: The highway_env environment instance.
    :param llm_agent: The LLMAgent instance for decision making.
    :param num_steps: Total simulation steps.
    :param env_video_file: Output path for the simulation video.
    :param output_csv_file: Output path for vehicle data CSV.
    :param fps: Frames per second for the output video.
    :param log_file_path: Path to the file where LLM reasoning logs will be saved.
    """
    env_frames = []
    
    action_to_int = {
        'LANE_LEFT': 0,
        'IDLE': 1,
        'LANE_RIGHT': 2,
        'FASTER': 3,
        'SLOWER': 4,
    }

    with open(output_csv_file, mode='w', newline='') as csvfile, \
         open(log_file_path, mode='w') as log_file: # Open log file
        
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed', 'llm_decision', 'reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Initial simulation setup logs
        print(f"--- Simulation Started ---")
        log_file.write(f"--- Simulation Started ---\n")
        print(f"Environment Setup: {NUM_MAIN_LANES} Main Lanes ({MAIN_LANE_INDICES}), Ramp Lane {RAMP_LANE_IDX}")
        log_file.write(f"Environment Setup: {NUM_MAIN_LANES} Main Lanes ({MAIN_LANE_INDICES}), Ramp Lane {RAMP_LANE_IDX}\n")
        print(f"Number of CAVs: {len(env.controlled_vehicles)}, Number of HDVs: {len(env.road.vehicles) - len(env.controlled_vehicles)}")
        log_file.write(f"Number of CAVs: {len(env.controlled_vehicles)}, Number of HDVs: {len(env.road.vehicles) - len(env.controlled_vehicles)}\n")
        print(f"Controlled CAVs IDs: {[x.id for x in env.controlled_vehicles]}")
        log_file.write(f"Controlled CAVs IDs: {[x.id for x in env.controlled_vehicles]}\n")

        for step in range(num_steps):
            print(f"\n--- Simulation Step {step} ---")
            log_file.write(f"\n--- Simulation Step {step} ---\n")
            
            current_step_cav_decisions: Dict[str, str] = {}
            
            decision_queue = PriorityQueue()
            actions_for_env = [1] * len(env.controlled_vehicles) # Initialize all controlled vehicles to IDLE

            # Populate priority queue for CAVs
            for vehicle in env.controlled_vehicles:
                priority_number = 0.0 
                temp_mock_vehicle_for_priority_check = MockVehicle(vehicle) 

                current_pos_x = vehicle.position[0] 
                merge_point_x = env.ends[2] 
                distance_to_merge_point = merge_point_x - current_pos_x

                if vehicle.lane_index[2] == RAMP_LANE_IDX: # Ramp lane (lane 2)
                    priority_number = -distance_to_merge_point 
                    priority_number -= 2000 
                elif vehicle.lane_index[2] == MERGE_MAIN_LANE_IDX: # Main lane involved in merge (lane 1)
                    priority_number = -distance_to_merge_point 
                    priority_number -= 1000 
                else: # Other main lanes (lane 0)
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

                original_controlled_veh_id_before_ego_set = getattr(controlled_veh_obj, 'id', f"vehicle_{id(controlled_veh_obj)}")
                
                controlled_veh_obj.id = "ego" 
                
                llm_agent.scenario._update_vehicles(current_step_cav_decisions) 

                # Print to console (concise) and log to file (full)
                print(f"  --- Processing CAV: Original ID '{original_controlled_veh_id_before_ego_set}', Temporarily 'ego' ---")
                log_file.write(f"  --- Processing CAV: Original ID '{original_controlled_veh_id_before_ego_set}', Temporarily 'ego' ---\n")
                
                if "ego" not in llm_agent.scenario.vehicles:
                    error_msg = f"    ERROR: 'ego' key NOT found in llm_agent.scenario.vehicles AFTER _update_vehicles! Scenario keys: {list(llm_agent.scenario.vehicles.keys())}"
                    print(error_msg)
                    log_file.write(error_msg + "\n")
                    llm_decision_output = llm_agent._fallback_decision(
                        {'ego_vehicle': {'lane_id': controlled_veh_obj.lane_index[2] if controlled_veh_obj.lane_index else -1}},
                        reason=f"KeyError: 'ego' in scenario.vehicles for original ID {original_controlled_veh_id_before_ego_set}"
                    )
                else:
                    current_ego_mock = llm_agent.scenario.vehicles["ego"]
                    print(f"    Found 'ego' in scenario.vehicles. Mock ID: {current_ego_mock.id}, Lane: {current_ego_mock.lane_idx}, Pos: {current_ego_mock.lanePosition:.2f}")
                    log_file.write(f"    Found 'ego' in scenario.vehicles. Mock ID: {current_ego_mock.id}, Lane: {current_ego_mock.lane_idx}, Pos: {current_ego_mock.lanePosition:.2f}\n")
                    
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

                    llm_decision_output = llm_agent.get_decision(observation_for_llm, log_file=log_file) # Pass log_file

                llm_action_str = llm_decision_output.get('decision', 'IDLE')
                reasoning = llm_decision_output.get('reasoning', 'No specific reasoning provided.')
                
                actions_for_env[idx_in_controlled_vehicles] = action_to_int.get(llm_action_str, 1) 
                
                current_step_cav_decisions[original_controlled_veh_id_before_ego_set] = llm_action_str

                writer.writerow({
                    'step': step,
                    'vehicle_id': original_controlled_veh_id_before_ego_set,
                    'lane_index': controlled_veh_obj.lane_index[2],
                    'position': controlled_veh_obj.position[0],
                    'speed': controlled_veh_obj.speed,
                    'llm_decision': llm_action_str,
                    'reasoning': reasoning
                })
                # Print to console (concise) and log to file (full)
                print(f"Step {step}: CAV '{original_controlled_veh_id_before_ego_set}' Final Decision: {llm_action_str} (Reasoning: {reasoning[:50]}...)")
                log_file.write(f"Step {step}: CAV '{original_controlled_veh_id_before_ego_set}' Final Decision: {llm_action_str}, Reasoning: {reasoning}\n")
            
                controlled_veh_obj.id = original_controlled_veh_id_before_ego_set
            
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)

            obs, reward, done, info = env.step(actions_for_env)
            
            print(f"Actual env actions applied by highway_env: {info.get('new_action', 'N/A')}")
            log_file.write(f"Actual env actions applied by highway_env: {info.get('new_action', 'N/A')}\n")

            if done:
                print(f"Simulation ended at step {step}")
                log_file.write(f"Simulation ended at step {step}\n")
                break

    print(f"\n--- Simulation Ended ---")
    log_file.write(f"\n--- Simulation Ended ---\n")
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    # 获取当前日期和时间，格式如 20240607_153012
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("./result", now_str)
    os.makedirs(result_dir, exist_ok=True)

    DEEPSEEK_API_KEY = "sk-d181f41df79741bca4b134881c718a9d"
    try:
        env = gym.make('merge-multi-agent-v0')
        env.reset()
        print("受控车辆 (CAVs):", [x.id for x in env.controlled_vehicles])
        print("路上所有车辆:", [x.id for x in env.road.vehicles])
        llm_agent = LLMAgent(api_key=DEEPSEEK_API_KEY, env=env, model_name="deepseek-coder")
        SIM_NUM_STEPS = 400
        SIM_FPS = 10
        ENV_VIDEO_FILE = os.path.join(result_dir, f"llm_merge_{now_str}.mp4")
        OUTPUT_CSV_FILE = os.path.join(result_dir, f"llm_merge_{now_str}.csv")
        LLM_REASONING_LOG_FILE = os.path.join(result_dir, f"llm_reasoning_log_{now_str}.txt")
        simulate_and_save_videos(env, llm_agent, 
                                 num_steps=SIM_NUM_STEPS, 
                                 env_video_file=ENV_VIDEO_FILE, 
                                 output_csv_file=OUTPUT_CSV_FILE, 
                                 fps=SIM_FPS,
                                 log_file_path=LLM_REASONING_LOG_FILE) 
        env.close()
    except Exception as e:
        import shutil
        shutil.rmtree(result_dir, ignore_errors=True)
        raise
