import openai
import json
import numpy as np
import re
import copy 
from typing import Any, Dict, List, Tuple, Optional

# Make sure to import Road from highway_env.road.road
from highway_env.road.road import Road 

# ==============================================================================
# 1. LLM Tools related definitions and classes
# ==============================================================================

class MockVehicle:
    """
    A mock vehicle class to adapt highway_env.Vehicle to Scenario's expected format.
    This helps the LLM tools to interpret vehicle data consistently.
    """
    def __init__(self, vehicle, decision: Optional[str] = None):
        self._vehicle = vehicle
        self.id = getattr(vehicle, 'id', f"vehicle_{id(vehicle)}")
        self.speed = vehicle.speed
        self.lane_id_tuple = vehicle.lane_index if vehicle.lane_index else (-1, -1, -1)
        self.lane_idx = vehicle.lane_index[2] if vehicle.lane_index else -1
        self.lanePosition = vehicle.position[0]
        self.original_lane_index = vehicle.lane_index
        self.is_controlled = hasattr(vehicle, 'is_controlled') and vehicle.is_controlled
        self.decision = decision

class Scenario:
    """
    Adapts highway_env.Env to the format expected by the LLM tools.
    This class wraps the environment state, providing a simplified view for LLM tool access.
    It holds the MockVehicle representation of all vehicles in the environment.
    """
    def __init__(self, env: Any):
        self.env = env # Keep reference to original environment
        self.vehicles: Dict[str, MockVehicle] = {} # This will be the source of truth for MockVehicles
        self.lanes: Dict[int, Any] = {} # Not directly used by tools in this setup, but can be kept for consistency
        self._update_vehicles() # Initial update
        self._update_lanes()

    def _update_vehicles(self, cav_current_step_decisions: Optional[Dict[str, str]] = None):
        """
        Populate the vehicles dictionary with MockVehicle instances, including current step decisions for CAVs.
        This Scenario's `vehicles` dictionary is the central place for tools to get MockVehicle data.
        :param cav_current_step_decisions: A dict mapping CAV IDs to their decisions made in the current simulation step.
        """
        if cav_current_step_decisions is None:
            cav_current_step_decisions = {}

        self.vehicles = {} # Clear previous vehicles
        
        # Add controlled vehicles, creating MockVehicle instances
        if self.env.controlled_vehicles:
            
            for v_orig in self.env.controlled_vehicles:
                # Use the original ID if it's not the "ego" being processed, else "ego"
                veh_id = getattr(v_orig, 'id', f"vehicle_{id(v_orig)}")
                decision_for_mock = cav_current_step_decisions.get(veh_id, None)
                mock_v = MockVehicle(v_orig, decision=decision_for_mock)
                mock_v.is_controlled = True
                self.vehicles[veh_id] = mock_v
        
        # Add other (non-controlled) vehicles
        for v_orig in self.env.road.vehicles:
            veh_id = getattr(v_orig, 'id', f"vehicle_{id(v_orig)}")
            if veh_id not in self.vehicles: # Avoid re-adding controlled vehicles if they share IDs for some reason
                mock_v = MockVehicle(v_orig)
                mock_v.is_controlled = False # Mark as uncontrolled/HDV
                
                self.vehicles[veh_id] = mock_v
        

    def _update_lanes(self):
        # This method's output is not directly used by current tools when they access env.road.network
        # but can be kept for conceptual completeness.
        self.lanes = {
            0: type('obj', (object,), {'laneIdx': 0})(),
            1: type('obj', (object,), {'laneIdx': 1})(),
            2: type('obj', (object,), {'laneIdx': 2})(),
            3: type('obj', (object,), {'laneIdx': 3})()
        }

# Helper function to check if a vehicle is in the merging area
def is_in_merging_area(vehicle: MockVehicle): # Type hint with MockVehicle
    return vehicle.original_lane_index in [("c", "d", 2), ("c", "d", 3)]

def prompts(name, description):
    def decorator(cls):
        cls.name = name
        cls.description = description
        return cls
    return decorator

# --- Action Mappings for LLM and Environment ---
ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'change lane to the left of the current lane,',
    1: 'remain in the current lane with current speed',
    2: 'change lane to the right of the current lane',
    3: 'accelerate the vehicle',
    4: 'decelerate the vehicle'
}

# --- Trajectory Prediction Constants ---
PREDICTION_TIMESTEP = 0.5
PREDICTION_HORIZON = 3.0
SAFE_COLLISION_DISTANCE = 3.0


def _predict_single_vehicle_trajectory(vehicle: 'MockVehicle', action: Optional[str], env_road: Road, env_dt: float) -> List[np.ndarray]:
    """
    Predicts the trajectory of a single vehicle given an action.
    This is a simplified prediction and does not account for complex
    interactions or low-level control details like lane changes.

    :param vehicle: The MockVehicle object to predict for.
    :param action: The high-level action (e.g., 'FASTER', 'LANE_LEFT').
    :param env_road: The highway_env Road object for lane info.
    :param env_dt: Simulation timestep (from env).
    :return: A list of future (x, y) positions.
    """
    future_positions = []
    
    # Create a deep copy of the original highway_env Vehicle object for prediction
    temp_veh = copy.deepcopy(vehicle._vehicle) 
    
    # Apply the action for prediction (mimic ControlledVehicle.act logic)
    if action == "FASTER":
        temp_veh.target_speed += temp_veh.DELTA_SPEED
    elif action == "SLOWER":
        temp_veh.target_speed -= temp_veh.DELTA_SPEED
    elif action == "LANE_LEFT":
        _from, _to, _id = temp_veh.target_lane_index
        # Get actual lane length to ensure clip is within bounds
        lane_len = len(env_road.network.graph[_from][_to])
        target_lane_index_for_prediction = _from, _to, np.clip(_id - 1, 0, lane_len - 1)
        if env_road.network.get_lane(target_lane_index_for_prediction).is_reachable_from(temp_veh.position):
            temp_veh.target_lane_index = target_lane_index_for_prediction
    elif action == "LANE_RIGHT":
        _from, _to, _id = temp_veh.target_lane_index
        lane_len = len(env_road.network.graph[_from][_to])
        target_lane_index_for_prediction = _from, _to, np.clip(_id + 1, 0, lane_len - 1)
        if env_road.network.get_lane(target_lane_index_for_prediction).is_reachable_from(temp_veh.position):
            temp_veh.target_lane_index = target_lane_index_for_prediction
    
    num_steps_prediction = int(PREDICTION_HORIZON / env_dt) # Use env_dt for granularity
    
    for _ in range(num_steps_prediction):
        temp_veh.act() 
        temp_veh.step(env_dt) 
        temp_veh.follow_road()
        future_positions.append(temp_veh.position)
        
    return future_positions


# --- Tool Classes ---

@prompts(name='Get_Available_Actions',
             description="""Useful to know what available actions (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER) the ego vehicle can take in this situation. Call this tool with input 'ego'.""")
class getAvailableActions:
    def __init__(self, scenario_instance: Scenario) -> None: # Tool now takes Scenario instance
        self.scenario = scenario_instance
    def inference(self, input: str) -> str:
        availableActions = list(ACTIONS_ALL.keys())
        outputPrefix = 'You can ONLY use one of the following actions: \n'
        for action_idx in availableActions:
            outputPrefix += f"{ACTIONS_ALL[action_idx]}--{ACTIONS_DESCRIPTION[action_idx]}; \n"
        outputPrefix += """\nTo check decision safety you should follow steps:
        Step 1: Get the vehicles in this lane that you may affect. Acceleration, deceleration and idle action affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Remember to use the proper tools mentioned in the tool list ONCE a time.
        """
        return outputPrefix

@prompts(name='Decision-making_Instructions',
             description="""This tool gives you general instructions on how to ensure an action is safe. The input to this tool should be a string, which is ONLY the action name for which you need safety instructions.""")
class isActionSafe:
    def __init__(self) -> None:
        pass
    def inference(self, action: str) -> str:
        return f"""To check action safety you should follow three steps:
        Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle action affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2:(Optional) Get the vehicles in this lane that you may affect, ONLY when you don't know.
        Step 3: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Follow the instructions and remember to use the proper tools mentioned in the tool list once a time.
        """

@prompts(name='Get_Available_Lanes',
             description="""Useful to know the available lanes (current, left, right) for a vehicle given the road network and merging rules. Input: vehicle ID (e.g., 'ego').""")
class getAvailableLanes:
    def __init__(self, scenario_instance: Scenario) -> None: # Tool now takes Scenario instance
        self.scenario = scenario_instance
        self.env = self.scenario.env # Access original env via scenario
    def inference(self, vid: str) -> str:
        ego_vehicle_obj = self.scenario.vehicles.get(vid) 
        if ego_vehicle_obj is None:
            return f"Vehicle with ID '{vid}' not found."
        current_lane_idx = ego_vehicle_obj.lane_idx
        available_lanes_info = [f"`lane_{current_lane_idx}` is the current lane."]
        
        current_lane_tuple = ego_vehicle_obj.lane_id_tuple
        side_lane_tuples = self.env.road.network.side_lanes(current_lane_tuple) # Use self.env.road.network
        
        for side_lane_tuple in side_lane_tuples:
            if side_lane_tuple[2] < current_lane_idx: # It's a left lane
                if current_lane_idx == 3 and side_lane_tuple[2] == 2 and not is_in_merging_area(ego_vehicle_obj):
                     available_lanes_info.append(f"Cannot change to `lane_2` from current lane `lane_3` as vehicle is not in merging area.")
                else:
                    available_lanes_info.append(f"`lane_{side_lane_tuple[2]}` is to the left of the current lane.")
            elif side_lane_tuple[2] > current_lane_idx: # It's a right lane
                if current_lane_idx == 2 and side_lane_tuple[2] == 3 and not is_in_merging_area(ego_vehicle_obj):
                    available_lanes_info.append(f"Cannot change to `lane_3` from current lane `lane_2` as vehicle is not in merging area.")
                else:
                    available_lanes_info.append(f"`lane_{side_lane_tuple[2]}` is to the right of the current lane.")
        
        return f"The available lanes of `{vid}` are: " + " ".join(available_lanes_info)


@prompts(name='Get_All_Nearby_Vehicles_Info',
             description="""Observes and returns detailed information about all nearby vehicles in the current lane, left lane, and right lane relative to the ego vehicle. This tool provides a comprehensive local perception. Input: 'ego'.""")
class Get_All_Nearby_Vehicles_Info:
    def __init__(self, scenario_instance: Scenario) -> None: # Tool now takes Scenario instance
        self.scenario = scenario_instance
        self.road = self.scenario.env.road # Access road via scenario's env
        self.network = self.road.network

    def inference(self, input: str) -> str: # Expects 'ego' as input
        if input != 'ego':
            return "Input for Get_All_Nearby_Vehicles_Info must be 'ego'."

        ego_mock_vehicle = self.scenario.vehicles.get("ego")
        if ego_mock_vehicle is None:
            return "Ego vehicle not found in scenario."

        current_lane_idx = ego_mock_vehicle.lane_idx
        current_lane_tuple = ego_mock_vehicle.lane_id_tuple 
        
        all_nearby_info_structured = {
            "current_lane": {"lane_id": f"lane_{current_lane_idx}", "front": [], "rear": []},
            "left_lane": None,
            "right_lane": None
        }

        side_lane_tuples = self.network.side_lanes(current_lane_tuple)
        
        relevant_lane_indices_map = {current_lane_idx: "current_lane"} # Map numerical idx to string key
        for sl_tuple in side_lane_tuples:
            side_idx = sl_tuple[2]
            if side_idx < current_lane_idx: # Left lane
                relevant_lane_indices_map[side_idx] = "left_lane"
                all_nearby_info_structured["left_lane"] = {"lane_id": f"lane_{side_idx}", "front": [], "rear": []}
            elif side_idx > current_lane_idx: # Right lane
                relevant_lane_indices_map[side_idx] = "right_lane"
                all_nearby_info_structured["right_lane"] = {"lane_id": f"lane_{side_idx}", "front": [], "rear": []}
        
        for v_id, v_mock in self.scenario.vehicles.items():
            if v_id == ego_mock_vehicle.id:
                continue 

            if v_mock.lane_idx in relevant_lane_indices_map and \
               self.network.is_connected_road(ego_mock_vehicle.original_lane_index, v_mock.original_lane_index, depth=1):
                
                distance_from_ego = v_mock.lanePosition - ego_mock_vehicle.lanePosition
                
                veh_detail = {
                    "id": v_mock.id,
                    "type": "CAV" if v_mock.is_controlled else "HDV",
                    "speed": round(v_mock.speed, 1), 
                    "distance_from_ego": round(distance_from_ego, 2), 
                    "lane_id": f"lane_{v_mock.lane_idx}", 
                    "current_decision_of_CAV": v_mock.decision if v_mock.is_controlled and v_mock.decision else "None"
                }

                lane_key = relevant_lane_indices_map[v_mock.lane_idx]
                if distance_from_ego > 0: all_nearby_info_structured[lane_key]["front"].append(veh_detail)
                else: all_nearby_info_structured[lane_key]["rear"].append(veh_detail)
        
        for lane_key in ["current_lane", "left_lane", "right_lane"]:
            if all_nearby_info_structured[lane_key] and isinstance(all_nearby_info_structured[lane_key], dict):
                all_nearby_info_structured[lane_key]["front"].sort(key=lambda x: x["distance_from_ego"])
                all_nearby_info_structured[lane_key]["rear"].sort(key=lambda x: -x["distance_from_ego"])

        return json.dumps(all_nearby_info_structured, indent=2)


@prompts(name='Check_Trajectory_Conflict',
             description="""Checks if the ego vehicle's planned action trajectory conflicts with a target vehicle's predicted trajectory.
             Input: comma-separated string 'ego_action,target_vehicle_id' (e.g., 'LANE_LEFT,vehicle_5' or 'FASTER,vehicle_8').""")
class CheckTrajectoryConflict:
    def __init__(self, scenario_instance: Scenario) -> None: # Tool now takes Scenario instance
        self.scenario = scenario_instance
        self.road = self.scenario.env.road 
        self.dt = 1 / self.scenario.env.config.get("simulation_frequency", 15) # Access dt via scenario.env.config

    def inference(self, inputs: str) -> str:
        try:
            ego_action_str, target_vid = inputs.replace(' ', '').split(',')
            if ego_action_str not in ACTIONS_ALL.values():
                return f"Invalid action string '{ego_action_str}'. Must be one of {list(ACTIONS_ALL.values())}."
        except ValueError:
            return "Invalid input format. Please provide 'ego_action,target_vehicle_id'."

        ego_mock_vehicle = self.scenario.vehicles.get("ego")
        target_mock_vehicle = self.scenario.vehicles.get(int(target_vid))

        if ego_mock_vehicle is None:
            return "Ego vehicle not found in scenario."
        if target_mock_vehicle is None:
            return f"Target vehicle '{target_vid}' not found in scenario."
        
        ego_trajectory = _predict_single_vehicle_trajectory(
            ego_mock_vehicle, ego_action_str, self.road, self.dt 
        )

        target_action_for_prediction = target_mock_vehicle.decision if target_mock_vehicle.is_controlled and target_mock_vehicle.decision else 'IDLE'
        
        if target_action_for_prediction not in ACTIONS_ALL.values():
            target_action_for_prediction = 'IDLE'

        target_trajectory = _predict_single_vehicle_trajectory(
            target_mock_vehicle, target_action_for_prediction, self.road, self.dt
        )
        
        min_distance = float('inf')
        collision_detected = False
        for i in range(min(len(ego_trajectory), len(target_trajectory))):
            dist = np.linalg.norm(ego_trajectory[i] - target_trajectory[i])
            if dist < min_distance:
                min_distance = dist
            if dist < SAFE_COLLISION_DISTANCE:
                collision_detected = True
                break

        if collision_detected:
            return (f"TRAJECTORY CONFLICT DETECTED between ego (action: {ego_action_str}) and {target_vid} "
                    f"(predicted action: {target_action_for_prediction}). "
                    f"Minimum predicted distance: {min_distance:.2f}m, which is less than safety threshold {SAFE_COLLISION_DISTANCE}m. "
                    f"Action {ego_action_str} is UNSAFE.")
        else:
            return (f"No trajectory conflict detected between ego (action: {ego_action_str}) and {target_vid} "
                    f"(predicted action: {target_action_for_prediction}). "
                    f"Minimum predicted distance: {min_distance:.2f}m, which is greater than safety threshold {SAFE_COLLISION_DISTANCE}m. "
                    f"Action {ego_action_str} is SAFE with {target_vid}.")


class LLMAgent:
    def __init__(self, api_key: str, env: Any, model_name: str = "deepseek-coder", temperature: float = 0.7):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name
        self.temperature = temperature
        self.env = env 
        self.scenario = Scenario(self.env) # LLMAgent owns the Scenario instance

        # Tools now receive scenario_instance directly
        self.tools = {
            getAvailableActions(self.scenario), # Pass scenario instance
            isActionSafe(), 
            getAvailableLanes(self.scenario), # Pass scenario instance
            Get_All_Nearby_Vehicles_Info(self.scenario), # Pass scenario instance
            CheckTrajectoryConflict(self.scenario), # Pass scenario instance
        }
        self.tools_map = {tool.name: tool for tool in self.tools}
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        tool_descriptions = ""
        for tool_obj in self.tools:
            tool_descriptions += f"""
            Tool Name: {tool_obj.name}
            Description: {tool_obj.description}
            """
        
        prompt = f"""
        You are an autonomous vehicle (CAV, identified as 'ego') operating on a highway merging scenario. Your goal is to select the safest and most efficient action based on your current position and surrounding traffic.

        --- Environment Setup ---
        - The highway has 3 main lanes (0 = leftmost, 1 = center, 2 = rightmost) and 1 on-ramp lane (3 = ramp).
        - The merging area is defined as segment ("c", "d") and specifically includes ("c", "d", 2) for main lane and ("c", "d", 3) for the ramp.
        - Vehicles can be HDV or CAV. You have full observation for all vehicles but cannot communicate with HDVs.
        - Each vehicle knows whether it is on the ramp or the main road, and whether it is in the merging zone.

        --- Lane Change Rules ---
        - **General:** Main lane 0 cannot LANE_LEFT; main lane 2 cannot LANE_RIGHT.
        - **Ramp Specific:**
            - **Ramp segments ("a", "b") and ("b", "c")**: Vehicles in these segments of the ramp (lane 3) can **only go straight** and **cannot change lanes**.
            - **Ramp segment ("c", "d")**: Vehicles in this segment of the ramp (lane 3) are in the **merging zone**. They **can perform LANE_LEFT** action to merge into main lane 2, if safe. The end of segment ("c", "d") is the terminal point of the ramp merging area.
        - **Main Lane Specific (Merging Zone):** Main road vehicles in the merging area (lane 2, segment ("c", "d")) can **consider LANE_LEFT** to assist ramp vehicles. Main road vehicles can never change right into the ramp.

        --- Action Space ---
        Your available actions are:
        - LANE_LEFT
        - LANE_RIGHT
        - FASTER
        - SLOWER
        - IDLE

        --- Available Tools ---
        You have access to the following specialized tools. You MUST use these tools to gather information about your environment and assess safety.
        {tool_descriptions}

        --- Decision Logic ---
        Your decision-making process will involve a series of tool calls and internal reasoning.
        1.  **Initial Perception:** Start by understanding your `ego_vehicle` basic information (current lane, position, speed, and whether you’re in the merging area) and `road_info`.
        2.  **Tool-Driven Information Gathering & Safety Assessment:** Systematically use tools to gather all necessary environmental perception and safety information. Prioritize steps:
            * First, call `Get_Available_Actions(input='ego')` to understand your action options.
            * Then, call `Get_Available_Lanes(input='ego')` to understand potential target lanes.
            * **Crucially, call `Get_All_Nearby_Vehicles_Info(input='ego')` to get a comprehensive overview of surrounding vehicles in all relevant lanes (current, left, right).** This tool's output will provide a structured JSON with 'current_lane', 'left_lane', 'right_lane' sections, each listing 'front' and 'rear' vehicles. For each vehicle, it will include its 'id', 'type' (CAV/HDV), 'speed', 'distance_from_ego', 'lane_id', and importantly, `current_decision_of_CAV` (if that CAV has already been processed in this step's priority queue), indicating their planned action in this step.
            * **After analyzing `Get_All_Nearby_Vehicles_Info`'s output, create a prioritized list of ego actions to evaluate.** For example, for ramp vehicles in merging zone, prioritize `LANE_LEFT`. For main lane vehicles, `IDLE` or `FASTER` might be prioritized, with `LANE_LEFT` for yielding as a secondary option.
            * For *each* prioritized ego action, and for *each identified nearby vehicle* (from `Get_All_Nearby_Vehicles_Info` output) that could be affected by that action, call `Check_Trajectory_Conflict(input='ego_action,target_vehicle_id')`. This tool will predict future trajectories for both 'ego' (with your proposed action) and the 'target_vehicle' (with its own decision if a CAV, or IDLE if HDV), and tell you if a collision is predicted.
            * **Safety is paramount.** If `Check_Trajectory_Conflict` indicates a conflict for any vehicle, that 'ego_action' is immediately deemed unsafe and must NOT be chosen. Discard it and move to the next prioritized action.
            * **HDV Behavior Prediction:** For HDVs, assume they will follow basic traffic rules and driving habits, typically maintaining their current lane and speed (IDLE) unless a specific action is predicted for them.
        3.  **Optimal Strategy Selection:** From the remaining safe actions, select the one that best meets the efficiency and comfort objectives.

        --- Merging Strategy (Crucial for Efficiency and Safety in Merging Zone) ---

        **Your overarching goal is to achieve successful and safe merging for all ramp vehicles while preventing collisions.**

        **Ramp vehicles (lane 3):**
        - **If in ramp segments ("a", "b") or ("b", "c")**: You CANNOT change lanes. You must continue straight (IDLE or adjust speed) to reach the merging zone.
        - **If in merging zone (ramp lane 3, segment "c", "d")**:
            - If it is safe to merge left into main lane 2 (as confirmed by tools), **PRIORITIZE LANE_LEFT**. This is your primary objective for successful merging.
            - Else, adjust speed (FASTER/SLOWER/IDLE) to seek or create gaps, actively looking for a safe opportunity to merge.
            - If at the very end of the ramp (end of segment "c", "d") with no safe gap to merge, choose SLOWER to avoid collision with the wall or main road vehicles.

        **Main road vehicles (lane 0-2):**
        - **If in merging zone (main lane 2, segment "c", "d")**:
            - If a ramp vehicle (from lane 3) needs to merge, and your left lane (lane 1 or 0) is safe and available, **CONSIDER LANE_LEFT to assist** in facilitating traffic flow for the merging vehicle.
            - Otherwise, adjust speed (FASTER/SLOWER/IDLE) to create a reasonable gap for the ramp vehicle to merge, but avoid heavy braking that disrupts main road flow.
        - In non-merging areas (any lane not in "c", "d" segment): drive efficiently but be prepared to facilitate merging if applicable.

        --- Output Format ---
        Return your final decision as a JSON object:
        {{
        "decision": "LANE_LEFT" | "LANE_RIGHT" | "FASTER" | "SLOWER" | "IDLE",
        "reasoning": "Explain clearly why this action is taken, based on your position, observations from tools (mention specific tool outputs), and adherence to the merging strategy. Explicitly state your vehicle type (ramp/main road) and if you are in the merging zone, and how that influenced your decision."
        }}

        If no safe option exists after all checks, return "SLOWER" and explain why all other options were unsafe.
        Your final response MUST be a valid JSON object.
        """
        return prompt

    def get_decision(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        ego_vehicle_info = observation['ego_vehicle']
        road_info = observation['road_info']
        traffic_density = observation.get('traffic_density', 'unknown')

        user_prompt_initial = f"""
        Current ego vehicle information: {json.dumps(ego_vehicle_info, indent=2)}
        Current road information: {json.dumps(road_info, indent=2)}
        Current traffic density: {traffic_density}
        
        Please begin your decision-making process by calling the appropriate tools to gather all necessary environmental perception and safety information.
        """
        messages.append({"role": "user", "content": user_prompt_initial})

        max_iterations = 15 

        for i in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=[
                        {"type": "function", "function": {"name": tool_obj.name, "description": tool_obj.description, "parameters": {"type": "object", "properties": {"input": {"type": "string"}}}}}
                        for tool_obj in self.tools
                    ],
                    tool_choice="auto", 
                    temperature=self.temperature,
                )
                
                response_message = response.choices[0].message
                
                # Print LLM's full message, then extract relevant info for concise logging
                # print(f"LLM Response (Iter {i+1}): {response_message.content[:80]}...") # Original full message print
                
                if response_message.tool_calls:
                    print(f"LLM Iter {i+1}: Calling Tools: {[tc.function.name for tc in response_message.tool_calls]}")
                else:
                    print(f"LLM Iter {i+1}: Text/Final: {response_message.content[:80]}...")
                
                messages.append(response_message)
                
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args_str = tool_call.function.arguments 
                        
                        tool_to_call = self.tools_map.get(function_name)
                        if tool_to_call:
                            try:
                                # Parse tool arguments: handle JSON string to dict, then get 'input'
                                parsed_args = json.loads(function_args_str)
                                tool_input = parsed_args.get('input') 
                                
                                tool_output_content = tool_to_call.inference(tool_input)
                                
                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_output_content,
                                })
                                # Concise print for tool output
                                brief_output = tool_output_content.replace('\n', ' ').strip()
                                if len(brief_output) > 100: brief_output = brief_output[:97] + "..."
                                print(f"  -> Tool Output: {function_name}: {brief_output}")

                            except json.JSONDecodeError:
                                error_msg = f"Invalid JSON args for {function_name}."
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(f"  -> Tool Error: {function_name}: {error_msg}")
                            except KeyError:
                                error_msg = f"Missing 'input' key for {function_name}."
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(error_msg)
                            except Exception as e:
                                error_msg = f"Error executing tool {function_name} with args {function_args_str}: {str(e)[:50]}..."
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(f"  -> Tool Error: {function_name}: {error_msg}")
                        else:
                            error_msg = f"Error: Tool '{function_name}' not found in tools_map."
                            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                            print(f"  -> Tool Error: {function_name}: {error_msg}")
                else:
                    # LLM provided a final answer (hopefully JSON) or a text response
                    llm_output_raw = response_message.content
                    
                    json_match = re.search(r'```json\n(.*?)```', llm_output_raw, re.DOTALL)
                    if json_match:
                        clean_llm_output = json_match.group(1).strip()
                    else:
                        clean_llm_output = llm_output_raw.strip()
                    
                    try:
                        decision = json.loads(clean_llm_output)
                        if all(k in decision for k in ["decision", "reasoning"]):
                            print(f"LLM Final Decision: {decision['decision']} (Reasoning: {decision['reasoning'][:50]}...)")
                            return decision 
                        else:
                            print(f"LLM output is JSON but missing required fields. Raw: {llm_output_raw[:80]}...")
                            messages.append({"role": "assistant", "content": llm_output_raw})
                            continue
                    except json.JSONDecodeError:
                        print(f"LLM did not return valid JSON. Raw: {llm_output_raw[:80]}... Appending as text and continuing.")
                        messages.append({"role": "assistant", "content": llm_output_raw})
                        continue

            except Exception as e:
                print(f"Error during LLM interaction (outer loop): {e}")
                return self._fallback_decision(observation, reason=f"An unexpected error occurred during LLM process: {e}")

        print(f"Max iterations ({max_iterations}) reached without LLM providing a valid JSON decision.")
        return self._fallback_decision(observation, reason=f"Max tool calls ({max_iterations}) reached, no valid JSON decision.")

    def _fallback_decision(self, observation: Dict[str, Any], reason: str = "LLM interaction failed, falling back to emergency deceleration.") -> Dict[str, Any]:
        print(reason)
        return {
            "decision": "SLOWER",
            "reasoning": reason
        }
