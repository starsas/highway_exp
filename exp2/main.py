import numpy as np
from moviepy.editor import ImageSequenceClip
import csv
import gym
import highway_env
import os
from llm_tools import LLMAgent, MockVehicle, ACTIONS_ALL, Scenario # Import necessary classes and functions from llm_tools

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
        # Removed 'llm_target_speed' from fieldnames
        fieldnames = ['step', 'vehicle_id', 'lane_index', 'position', 'speed', 'llm_decision', 'llm_target_lane', 'reasoning']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(num_steps):
            # Update the scenario for the current environment state for all tools
            # Note: llm_agent.scenario._update_vehicles() is implicitly handled by passing controlled_veh_obj as ego
            # for each query. This ensures tools have the latest full road vehicle state.
            
            actions_for_env = [1] * len(env.controlled_vehicles) # Initialize all controlled vehicles to IDLE
            
            # Iterate through each controlled vehicle to get a decision from the LLM
            # The LLM's 'ego' is dynamically set to the current controlled vehicle being processed.
            for idx, controlled_veh_obj in enumerate(env.controlled_vehicles):
                
                # To make the LLM reason about this specific controlled vehicle as "ego",
                # we update the MockVehicle ID within the llm_agent's internal scenario.
                # This is a bit of a hack, but it works with the current tool design.
                # A cleaner solution might involve passing ego_vehicle_obj directly to tools,
                # or having Scenario._update_vehicles take a specific ego ID.
                # For brevity and current structure:
                original_ego_id = None
                if "ego" in llm_agent.scenario.vehicles:
                    original_ego_id = llm_agent.scenario.vehicles["ego"]._vehicle.id
                
                # Temporarily make current controlled_veh_obj the "ego" in the scenario
                # Ensure the actual vehicle object's ID is set to "ego" for tools to find it.
                temp_original_id = getattr(controlled_veh_obj, 'id', f"vehicle_{id(controlled_veh_obj)}")
                controlled_veh_obj.id = "ego" 
                llm_agent.scenario._update_vehicles() # Update internal mock vehicles with new ego assignment

                # Now, construct observation for this "ego" vehicle
                current_ego_mock = llm_agent.scenario.vehicles["ego"] # Retrieve the newly set "ego" mock vehicle
                
                nearby_vehicles = []
                for v_road in env.road.vehicles:
                    if v_road.id != current_ego_mock.id: # Exclude the current "ego" vehicle itself
                        nearby_v_mock = MockVehicle(v_road)
                        nearby_vehicles.append({
                            'id': nearby_v_mock.id,
                            'speed': nearby_v_mock.speed,
                            'lane_id': nearby_v_mock.lane_id,
                            'lanePosition': nearby_v_mock.lanePosition,
                            'position_xy': v_road.position.tolist()
                        })
                
                # Sort nearby vehicles by distance to the current controlled vehicle
                nearby_vehicles.sort(key=lambda x: np.linalg.norm(np.array(x['position_xy']) - controlled_veh_obj.position))

                observation = {
                    'ego_vehicle': {
                        'id': current_ego_mock.id,
                        'speed': current_ego_mock.speed,
                        'lane_id': current_ego_mock.lane_id,
                        'lanePosition': current_ego_mock.lanePosition,
                        'position_xy': controlled_veh_obj.position.tolist(),
                        'in_merging_area': is_in_merging_area(current_ego_mock) # Provide this explicitly
                    },
                    'nearby_vehicles': nearby_vehicles,
                    'road_info': {
                        'ends': env.ends.tolist(),
                        'num_lanes': len(env.road.network.graph['a']['b']) # Example, adjust if needed
                    },
                    'traffic_density': env.config.get("traffic_density", "unknown")
                }

                # Get decision from LLM agent for the current controlled vehicle
                llm_decision_output = llm_agent.get_decision(observation)
                
                llm_action_str = llm_decision_output.get('decision', 'IDLE')
                target_lane = llm_decision_output.get('target_lane', controlled_veh_obj.lane_index[2] if controlled_veh_obj.lane_index else -1)
                reasoning = llm_decision_output.get('reasoning', 'No specific reasoning provided.')
                
                # Map LLM's string action to environment's integer action
                actions_for_env[idx] = action_to_int.get(llm_action_str, 1) # Default to IDLE if unknown

                # Log data for the current controlled vehicle
                writer.writerow({
                    'step': step,
                    'vehicle_id': temp_original_id, # Log original ID
                    'lane_index': controlled_veh_obj.lane_index[2],
                    'position': controlled_veh_obj.position[0],
                    'speed': controlled_veh_obj.speed,
                    'llm_decision': llm_action_str,
                    'llm_target_lane': target_lane,
                    'reasoning': reasoning
                })
                print(f"Step: {step}, Controlled Vehicle ID: {temp_original_id}, Decision (LLM): {llm_action_str}, Target Lane: {target_lane}, Reasoning: {reasoning}")
            
                # Revert vehicle ID for next iteration or for environment integrity
                controlled_veh_obj.id = temp_original_id
            
            # Render environment and record frame
            env_frame = env.render(mode='rgb_array')
            env_frames.append(env_frame)

            # Perform step for all controlled vehicles
            obs, reward, done, info = env.step(actions_for_env)
            
            # The 'new_action' in info might reflect internal adjustments by highway_env (e.g., safety overrides)
            print(f"Actual env actions applied by highway_env: {info.get('new_action', 'N/A')}")

            if done:
                print(f"Simulation ended at step {step}")
                break

    # Save video
    env_clip = ImageSequenceClip(env_frames, fps=fps)
    env_clip.write_videofile(env_video_file, codec="libx264")
    print(f"Environment video saved as {env_video_file}")

if __name__ == "__main__":
    # Initialize DeepSeek API Key from environment variables
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please set it using: export DEEPSEEK_API_KEY='YOUR_API_KEY'")
        exit()

    # Create the environment
    env = gym.make('merge-multi-agent-v0')
    env.reset()

    print("Controlled Vehicles (CAVs):", [x.id for x in env.controlled_vehicles])
    print("All Vehicles on Road:", [x.id for x in env.road.vehicles])

    # Initialize LLM Agent
    llm_agent = LLMAgent(api_key=DEEPSEEK_API_KEY, env=env, model_name="deepseek-coder")

    # Run simulation with LLM agent
    simulate_with_llm_agent(env, llm_agent, num_steps=400, env_video_file="./llm_merge_control.mp4", output_csv_file="llm_merge_control.csv", fps=10)
    env.close()