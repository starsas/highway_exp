import numpy as np
from moviepy.editor import ImageSequenceClip
import csv
import gym
import highway_env
import os
from queue import PriorityQueue 
# Make sure your custom_llm.py is correctly named and importable
from complex_llm import LLMAgent, MockVehicle, ACTIONS_ALL, Scenario, is_in_merging_area 
import re # Not strictly needed if LLM output is guaranteed pure JSON at end, but good for robust parsing

def simulate_with_llm_agent(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control.mp4", output_csv_file="llm_merge_control.csv", fps=10):
    env_frames = []
    
    action_to_int = {
        'LANE_LEFT': 0,
        'IDLE': 1,
        'LANE_RIGHT': 2,
        'FASTER': 3,
        'SLOWER': 4,
    }

    with open(output_csv_file, mode='w', newline='') as csvfile:
        # Changed fieldnames: removed 'llm_target_lane'
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed', 'llm_decision', 'reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(num_steps):
            # Update the scenario for the current environment state for all tools
            # This is crucial because LLM agent's tools will query `llm_agent.scenario.vehicles` directly.
            llm_agent.scenario._update_vehicles() 
            
            actions_for_env = [1] * len(env.controlled_vehicles) # Initialize all controlled vehicles to IDLE
            
            decision_queue = PriorityQueue()

            for controlled_veh_obj in env.controlled_vehicles:
                priority_number = 0.0 
                
                temp_mock_vehicle = MockVehicle(controlled_veh_obj) 
                in_merge_area_flag = is_in_merging_area(temp_mock_vehicle)

                current_pos_x = controlled_veh_obj.position[0] 
                
                merge_point_x = env.ends[2] 
                
                distance_to_merge_point = merge_point_x - current_pos_x

                if controlled_veh_obj.lane_index[2] == 3: # Ramp lane (lane 3)
                    priority_number = -distance_to_merge_point 
                    priority_number -= 1000 

                elif controlled_veh_obj.lane_index[2] == 2: # Main lane next to merge (lane 2)
                    priority_number = -distance_to_to_merge_point

                else: # Other main lanes (lane 0, 1)
                    priority_number = 10000 + current_pos_x 

                priority_number += np.random.rand() * 0.001 
                
                decision_queue.put((priority_number, controlled_veh_obj))

            processed_controlled_vehicles_ids = [] 
            while not decision_queue.empty():
                priority, controlled_veh_obj = decision_queue.get()
                
                if controlled_veh_obj.id in processed_controlled_vehicles_ids:
                    continue
                processed_controlled_vehicles_ids.append(controlled_veh_obj.id)

                idx = env.controlled_vehicles.index(controlled_veh_obj)

                # Temporarily make current controlled_veh_obj the "ego" in the scenario
                temp_original_id = getattr(controlled_veh_obj, 'id', f"vehicle_{id(controlled_veh_obj)}")
                controlled_veh_obj.id = "ego" 
                # Note: llm_agent.scenario._update_vehicles() is already called ONCE at the beginning of the step
                # No need to call it inside this loop, as the Scenario instance already holds all real vehicles.
                # The "ego" ID assignment to controlled_veh_obj is enough for tools to find it.

                # Construct observation for LLM (ONLY ego_vehicle_info and road_info)
                # LLM will use tools to get 'nearby_vehicles'
                current_ego_mock = llm_agent.scenario.vehicles["ego"] # Access the mock vehicle in scenario
                
                observation_for_llm = {
                    'ego_vehicle': {
                        'id': current_ego_mock.id,
                        'speed': current_ego_mock.speed,
                        'lane_id': current_ego_mock.lane_id,
                        'lanePosition': current_ego_mock.lanePosition,
                        'position_xy': controlled_veh_obj.position.tolist(), # Real vehicle position
                        'in_merging_area': is_in_merging_area(current_ego_mock)
                    },
                    'road_info': {
                        'ends': env.ends, 
                        'num_lanes': len(env.road.network.graph['a']['b'])
                    },
                    'traffic_density': env.config.get("traffic_density", "unknown")
                }

                # Get decision from LLM agent for the current controlled vehicle
                llm_decision_output = llm_agent.get_decision(observation_for_llm)
                
                llm_action_str = llm_decision_output.get('decision', 'IDLE')
                # target_lane is no longer directly returned by LLM, need to infer if needed for action.
                # For LANE_LEFT/LANE_RIGHT actions, highway_env will determine the target lane.
                # For IDLE/FASTER/SLOWER, the target lane remains current lane.
                
                reasoning = llm_decision_output.get('reasoning', 'No specific reasoning provided.')
                
                actions_for_env[idx] = action_to_int.get(llm_action_str, 1) 

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
            
                controlled_veh_obj.id = temp_original_id
            
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)

            obs, reward, done, info = env.step(actions_for_env)
            
            print(f"Actual env actions applied by highway_env: {info.get('new_action', 'N/A')}")

            if done:
                print(f"Simulation ended at step {step}")
                break

    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    DEEPSEEK_API_KEY = "sk-d181f41df79741bca4b134881c718a9d" # 你的实际API Key

    env = gym.make('merge-multi-agent-v0')
    env.reset()

    llm_agent = LLMAgent(api_key=DEEPSEEK_API_KEY, env=env, model_name="deepseek-coder")

    simulate_with_llm_agent(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control.mp4", output_csv_file="llm_merge_control.csv", fps=10)
    env.close()