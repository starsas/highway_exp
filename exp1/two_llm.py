import openai
import json
import numpy as np
import re
import copy 
from typing import Any, Dict, List, Tuple, Optional

# Make sure to import Road from highway_env.road.road
from highway_env.road.road import Road 
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle # Import relevant HDV types
from highway_env.vehicle.controller import ControlledVehicle # Import base CAV type

# --- Constants for Lane/Road Structure (Reflect the new 2 main + 1 ramp model) ---
NUM_MAIN_LANES = 2 # Number of main lanes: lane 0, lane 1
MAIN_LANE_INDICES = [0, 1]
RAMP_LANE_IDX = 2 # The ramp is now explicitly lane 2
MERGE_MAIN_LANE_IDX = 1 # Main lane involved in merge is lane 1 (rightmost main lane)
MERGE_RAMP_LANE_IDX = 2 # Ramp lane involved in merge is lane 2

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
        self.is_controlled = isinstance(vehicle, ControlledVehicle) # More robust check for controlled type
        self.decision = decision

class Scenario:
    """
    Adapts highway_env.Env to the format expected by the LLM tools.
    This class holds the MockVehicle representation of all vehicles in the environment.
    """
    def __init__(self, env: Any):
        self.env = env 
        self.vehicles: Dict[str, MockVehicle] = {}
        self.lanes: Dict[int, Any] = {}
        self._update_vehicles()
        self._update_lanes()

    def _update_vehicles(self, cav_current_step_decisions: Optional[Dict[str, str]] = None):
        if cav_current_step_decisions is None:
            cav_current_step_decisions = {}

        self.vehicles = {}
        
        if self.env.controlled_vehicles:
            for v_orig in self.env.controlled_vehicles:
                veh_id = getattr(v_orig, 'id', f"vehicle_{id(v_orig)}")
                decision_for_mock = cav_current_step_decisions.get(veh_id, None)
                mock_v = MockVehicle(v_orig, decision=decision_for_mock)
                mock_v.is_controlled = True
                self.vehicles[veh_id] = mock_v
        
        for v_orig in self.env.road.vehicles:
            veh_id = getattr(v_orig, 'id', f"vehicle_{id(v_orig)}")
            if veh_id not in self.vehicles:
                mock_v = MockVehicle(v_orig)
                mock_v.is_controlled = False
                self.vehicles[veh_id] = mock_v

    def _update_lanes(self):
        self.lanes = {}
        for i in range(NUM_MAIN_LANES + 1): 
            self.lanes[i] = type('obj', (object,), {'laneIdx': i})()

# Helper function to check if a vehicle is in the merging area (Updated for new lane IDs)
def is_in_merging_area(vehicle: MockVehicle):
    return vehicle.original_lane_index in [("c", "d", MERGE_MAIN_LANE_IDX), ("c", "d", MERGE_RAMP_LANE_IDX)]

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
SAFE_COLLISION_DISTANCE = 15.0 # Minimum distance between centers where collision is detected in prediction

# --- New Constant for Nearby Vehicle Filtering ---
NEARBY_OBSERVATION_DISTANCE = 30.0 # [m] Max distance to observe vehicles in adjacent lanes

# --- New Constants for Following Distance Pre-check (for FASTER/IDLE) ---
MIN_FOLLOWING_DISTANCE_FASTER = 25.0 # [m] Minimum current longitudinal distance for FASTER action with front car
MIN_FOLLOWING_DISTANCE_IDLE = 15.0 # [m] Minimum current longitudinal distance for IDLE action with front car

# --- New Constant for Aggressive Lane Change Pre-check ---
CRITICAL_LANE_CHANGE_DISTANCE_LONGITUDINAL = 15.0 # [m] If current longitudinal distance is within this, lane change is immediately unsafe


def _predict_single_vehicle_trajectory(vehicle: 'MockVehicle', action: Optional[str], env_road: Road, env_dt: float) -> List[np.ndarray]:
    future_positions = []
    
    # Create a deep copy of the original highway_env Vehicle object for prediction
    temp_veh = copy.deepcopy(vehicle._vehicle) 
    
    # --- IMPORTANT: Custom prediction logic for HDV (IDMVehicle) ---
    # If the vehicle is an HDV (IDMVehicle or LinearVehicle) and its predicted action is 'IDLE'
    # (which is the default for HDVs), we allow its act() method to simulate its actual IDM behavior
    # which includes reacting to a front vehicle. This makes HDV prediction more realistic.
    # Otherwise, if it's a CAV or an HDV with a specific non-IDLE action,
    # we simulate it as a ControlledVehicle to ensure target_speed/lane is followed.
    if isinstance(temp_veh, (IDMVehicle, LinearVehicle)) and action == 'IDLE':
        # Let the original HDV act() method dictate its behavior
        # No need to convert to ControlledVehicle if it's expected to follow its own internal model.
        pass # temp_veh remains its original type (e.g., IDMVehicle)
    else:
        # For CAVs, or HDVs if we want to force them to follow a specific non-IDLE action,
        # ensure it behaves like a ControlledVehicle.
        initial_speed = temp_veh.speed
        initial_target_lane_index = temp_veh.lane_index
        initial_route = temp_veh.route

        temp_veh = ControlledVehicle(temp_veh.road, temp_veh.position, temp_veh.heading, initial_speed, initial_target_lane_index, initial_speed, initial_route)
        temp_veh.target_speed = initial_speed # Ensure target speed is set to initial speed for the base case

    # Apply the action for prediction (mimic ControlledVehicle.act logic for CAVs or forced behavior)
    if action == "FASTER":
        temp_veh.target_speed += temp_veh.DELTA_SPEED
    elif action == "SLOWER":
        temp_veh.target_speed -= temp_veh.DELTA_SPEED
    elif action == "LANE_LEFT":
        _from, _to, _id = temp_veh.target_lane_index
        lane_len = 0
        if _from in env_road.network.graph and _to in env_road.network.graph[_from]:
            lane_len = len(env_road.network.graph[_from][_to])
        lane_len = max(1, lane_len) 
        
        target_lane_index_for_prediction = _from, _to, np.clip(_id - 1, 0, lane_len - 1)
        if env_road.network.get_lane(target_lane_index_for_prediction).is_reachable_from(temp_veh.position):
            temp_veh.target_lane_index = target_lane_index_for_prediction
    elif action == "LANE_RIGHT":
        _from, _to, _id = temp_veh.target_lane_index
        lane_len = 0
        if _from in env_road.network.graph and _to in env_road.network.graph[_from]:
            lane_len = len(env_road.network.graph[_from][_to])
        lane_len = max(1, lane_len)

        target_lane_index_for_prediction = _from, _to, np.clip(_id + 1, 0, lane_len - 1)
        if env_road.network.get_lane(target_lane_index_for_prediction).is_reachable_from(temp_veh.position):
            temp_veh.target_lane_index = target_lane_index_for_prediction
    
    num_steps_prediction = int(PREDICTION_HORIZON / env_dt) 
    
    for _ in range(num_steps_prediction):
        temp_veh.act() # This will call the appropriate act() method (IDM or Controlled)
        temp_veh.step(env_dt) 
        temp_veh.follow_road()
        future_positions.append(temp_veh.position)
        
    return future_positions


# --- Tool Classes ---

@prompts(name='Get_Available_Actions',
             description="""Useful to know what available actions (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER) the ego vehicle can take in this situation. Call this tool with input 'ego'.""")
class getAvailableActions:
    def __init__(self, scenario_instance: Scenario) -> None:
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
    def __init__(self, scenario_instance: Scenario) -> None:
        self.scenario = scenario_instance
        self.env = self.scenario.env
    def inference(self, vid: str) -> str:
        ego_vehicle_obj = self.scenario.vehicles.get(vid) 
        if ego_vehicle_obj is None:
            return f"Vehicle with ID '{vid}' not found."
        current_lane_idx = ego_vehicle_obj.lane_idx
        available_lanes_info = [f"`lane_{current_lane_idx}` is the current lane.`"] 
        
        current_lane_tuple = ego_vehicle_obj.lane_id_tuple
        side_lane_tuples = self.env.road.network.side_lanes(current_lane_tuple)
        
        for side_lane_tuple in side_lane_tuples:
            side_idx = side_lane_tuple[2]
            if side_idx < current_lane_idx: # It's a left lane
                if current_lane_idx == RAMP_LANE_IDX and side_idx == MERGE_MAIN_LANE_IDX and not is_in_merging_area(ego_vehicle_obj):
                     available_lanes_info.append(f"Cannot change to `lane_{MERGE_MAIN_LANE_IDX}` from current lane `lane_{RAMP_LANE_IDX}` as vehicle is not in merging area.")
                elif current_lane_idx == MERGE_MAIN_LANE_IDX and side_idx == MAIN_LANE_INDICES[0]: # Main lane 1 to main lane 0
                     available_lanes_info.append(f"`lane_{side_idx}` is to the left of the current lane.")
                else: # Generic left lane
                    available_lanes_info.append(f"`lane_{side_idx}` is to the left of the current lane.")
            elif side_idx > current_lane_idx: # It's a right lane
                if current_lane_idx == MERGE_MAIN_LANE_IDX and side_idx == RAMP_LANE_IDX and not is_in_merging_area(ego_vehicle_obj):
                    available_lanes_info.append(f"Cannot change to `lane_{RAMP_LANE_IDX}` from current lane `lane_{MERGE_MAIN_LANE_IDX}` as vehicle is not in merging area.")
                else: # Generic right lane
                    available_lanes_info.append(f"`lane_{side_idx}` is to the right of the current lane.")
        
        return f"The available lanes of `{vid}` are: " + " ".join(available_lanes_info)


@prompts(name='Get_All_Nearby_Vehicles_Info',
             description=f"""Observes and returns a list of dictionaries about all nearby vehicles. Each dictionary contains:
             'id': Vehicle ID (int or str)
             'type': 'CAV' or 'HDV'
             'lane_id': Lane ID (e.g., 'lane_0', 'lane_1', 'lane_2')
             'distance_from_ego': Distance to ego (positive for front, negative for rear).
             For the current lane, only the single closest vehicle in front and behind are included.
             For left/right adjacent lanes (if they exist), vehicles within {NEARBY_OBSERVATION_DISTANCE}m front/rear of ego are included.""")
class Get_All_Nearby_Vehicles_Info:
    def __init__(self, scenario_instance: Scenario) -> None:
        self.scenario = scenario_instance
        self.road = self.scenario.env.road
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
            "left_lane": {"lane_id": None, "front": [], "rear": []}, 
            "right_lane": {"lane_id": None, "front": [], "rear": []}
        }
        
        side_lane_tuples = self.network.side_lanes(current_lane_tuple) 
        
        relevant_lane_indices_map = {current_lane_idx: "current_lane"} 
        for sl_tuple in side_lane_tuples:
            side_idx = sl_tuple[2]
            if side_idx < current_lane_idx: # Left lane
                relevant_lane_indices_map[side_idx] = "left_lane"
                all_nearby_info_structured["left_lane"] = {"lane_id": f"lane_{side_idx}", "front": [], "rear": []}
            elif side_idx > current_lane_idx: # Right lane
                relevant_lane_indices_map[side_idx] = "right_lane"
                all_nearby_info_structured["right_lane"] = {"lane_id": f"lane_{side_idx}", "front": [], "rear": []}
        
        # Temporary lists to find closest front/rear in current lane
        current_lane_front_candidates = []
        current_lane_rear_candidates = []

        for v_id, v_mock in self.scenario.vehicles.items():
            if v_id == ego_mock_vehicle.id:
                continue 

            if v_mock.lane_idx in relevant_lane_indices_map and \
               self.network.is_connected_road(ego_mock_vehicle.original_lane_index, v_mock.original_lane_index, depth=1):
                
                distance_from_ego = v_mock.lanePosition - ego_mock_vehicle.lanePosition
                
                veh_detail = {
                    "id": v_mock.id,
                    "type": "CAV" if v_mock.is_controlled else "HDV",
                    "speed": round(v_mock.speed, 1), # m/s
                    "distance_from_ego": round(distance_from_ego, 2), # positive for front, negative for rear
                    "lane_id": f"lane_{v_mock.lane_idx}", # Specific lane_id of this vehicle
                    "current_decision_of_CAV": v_mock.decision if v_mock.is_controlled and v_mock.decision else "None" # Decision of other CAVs
                }

                lane_key = relevant_lane_indices_map[v_mock.lane_idx]
                
                if lane_key == "current_lane":
                    # Only single closest front/rear for current lane
                    if distance_from_ego > 0: current_lane_front_candidates.append(veh_detail)
                    else: current_lane_rear_candidates.append(veh_detail)
                else: # For left_lane or right_lane
                    if abs(distance_from_ego) <= NEARBY_OBSERVATION_DISTANCE:
                        if distance_from_ego > 0: all_nearby_info_structured[lane_key]["front"].append(veh_detail)
                        else: all_nearby_info_structured[lane_key]["rear"].append(veh_detail)
        
        # Process current lane: add only closest front and rear
        if current_lane_front_candidates:
            current_lane_front_candidates.sort(key=lambda x: x["distance_from_ego"])
            all_nearby_info_structured["current_lane"]["front"].append(current_lane_front_candidates[0])
        if current_lane_rear_candidates:
            current_lane_rear_candidates.sort(key=lambda x: -x["distance_from_ego"]) # Closest (smallest abs dist) first
            all_nearby_info_structured["current_lane"]["rear"].append(current_lane_rear_candidates[0])

        # Sort all collected vehicles for consistent output (e.g., by lane_id then by distance_from_ego)
        for lane_key in ["current_lane", "left_lane", "right_lane"]:
            if all_nearby_info_structured[lane_key] and isinstance(all_nearby_info_structured[lane_key], dict):
                all_nearby_info_structured[lane_key]["front"].sort(key=lambda x: x["distance_from_ego"])
                all_nearby_info_structured[lane_key]["rear"].sort(key=lambda x: -x["distance_from_ego"])
            elif all_nearby_info_structured[lane_key] is not None and all_nearby_info_structured[lane_key]["lane_id"] is None:
                 all_nearby_info_structured[lane_key] = "Lane does not exist relative to ego or is not reachable."
        
        return json.dumps(all_nearby_info_structured, indent=2)


@prompts(name='Check_Trajectory_Conflict',
             description="""Checks if the ego vehicle's planned action trajectory conflicts with a target vehicle's predicted trajectory.
             Input: comma-separated string 'ego_action,target_vehicle_id' (e.g., 'LANE_LEFT,vehicle_5' or 'FASTER,vehicle_8').""")
class CheckTrajectoryConflict:
    def __init__(self, scenario_instance: Scenario) -> None:
        self.scenario = scenario_instance
        self.road = self.scenario.env.road 
        self.dt = 1 / self.scenario.env.config.get("simulation_frequency", 15)

    def inference(self, inputs: str) -> str:
        try:
            ego_action_str, target_vid_str = inputs.replace(' ', '').split(',') 
            if ego_action_str not in ACTIONS_ALL.values():
                return f"Invalid action string '{ego_action_str}'. Must be one of {list(ACTIONS_ALL.values())}."
        except ValueError:
            return "Invalid input format. Please provide 'ego_action,target_vehicle_id'."

        ego_mock_vehicle = self.scenario.vehicles.get("ego")
        
        target_vid_key = target_vid_str
        if isinstance(target_vid_str, str) and target_vid_str.isdigit():
            target_vid_key = int(target_vid_str)
            
        target_mock_vehicle = self.scenario.vehicles.get(target_vid_key)

        if ego_mock_vehicle is None:
            return "Ego vehicle not found in scenario."
        
        if target_mock_vehicle is None:
            return f"TARGET_VEHICLE_NOT_FOUND: Vehicle '{target_vid_str}' was not found in the current environment. This might mean it despawned or crashed. Consider it safe if your action was based on its presence, or re-evaluate your plan if its absence changes critical factors."
        
        # --- NEW PRE-CHECK FOR LANE CHANGES AND FOLLOWING DISTANCE ---
        longitudinal_distance = target_mock_vehicle.lanePosition - ego_mock_vehicle.lanePosition
        
        # Determine if target_mock_vehicle is a front vehicle or rear vehicle
        is_front_vehicle = longitudinal_distance > 0
        
        if ego_action_str in ['LANE_LEFT', 'LANE_RIGHT']:
            # Lane Change Pre-check (applies to target lane, not current lane)
            # This check is more about if there's enough space to even begin the maneuver
            if is_front_vehicle: # If checking front vehicle in target lane
                if longitudinal_distance < SAFE_LANE_CHANGE_DISTANCE_LONGITUDINAL:
                    return (f"LANE_CHANGE_PRECHECK_UNSAFE: Action {ego_action_str} would put ego too close "
                            f"to front vehicle {target_vid_str} (dist: {longitudinal_distance:.2f}m) in target lane. "
                            f"Longitudinal distance less than {SAFE_LANE_CHANGE_DISTANCE_LONGITUDINAL}m. Action {ego_action_str} is UNSAFE.")
            else: # If checking rear vehicle in target lane
                if abs(longitudinal_distance) < SAFE_LANE_CHANGE_DISTANCE_LONGITUDINAL * 0.7: # Rear buffer can be slightly less strict
                    return (f"LANE_CHANGE_PRECHECK_UNSAFE: Action {ego_action_str} would cut off "
                            f"rear vehicle {target_vid_str} (dist: {abs(longitudinal_distance):.2f}m) in target lane. "
                            f"Longitudinal distance less than {SAFE_LANE_CHANGE_DISTANCE_LONGITUDINAL * 0.7:.2f}m. Action {ego_action_str} is UNSAFE.")
        elif ego_action_str == 'FASTER':
            # FASTER action pre-check: If front vehicle is too close, accelerating is unsafe.
            if is_front_vehicle and longitudinal_distance < MIN_FOLLOWING_DISTANCE_FASTER:
                 return (f"FOLLOWING_PRECHECK_UNSAFE: Action {ego_action_str} with {target_vid_str} (dist: {longitudinal_distance:.2f}m) is unsafe. "
                         f"Current front distance is less than {MIN_FOLLOWING_DISTANCE_FASTER}m. Action {ego_action_str} is UNSAFE.")
        elif ego_action_str == 'IDLE':
            # IDLE action pre-check: If front vehicle is too close, maintaining speed might be unsafe.
            if is_front_vehicle and longitudinal_distance < MIN_FOLLOWING_DISTANCE_IDLE:
                 return (f"FOLLOWING_PRECHECK_UNSAFE: Action {ego_action_str} with {target_vid_str} (dist: {longitudinal_distance:.2f}m) is unsafe. "
                         f"Current front distance is less than {MIN_FOLLOWING_DISTANCE_IDLE}m. Action {ego_action_str} is UNSAFE.")
        # --- END NEW PRE-CHECK ---

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
            # Adjust collision condition for FASTER action, to be more conservative
            if ego_action_str == 'FASTER':
                # If accelerating, be more strict: check against 1.5 * SAFE_COLLISION_DISTANCE
                if dist < SAFE_COLLISION_DISTANCE * 1.5: # Example: 15m * 1.5 = 22.5m
                    collision_detected = True
                    break
            # General collision check
            if dist < SAFE_COLLISION_DISTANCE:
                collision_detected = True
                break

        if collision_detected:
            return (f"TRAJECTORY CONFLICT DETECTED between ego (action: {ego_action_str}) and {target_vid_str} "
                    f"(predicted action: {target_action_for_prediction}). "
                    f"Minimum predicted distance: {min_distance:.2f}m, which is less than safety threshold {SAFE_COLLISION_DISTANCE}m. "
                    f"Action {ego_action_str} is UNSAFE.")
        else:
            return (f"No trajectory conflict detected between ego (action: {ego_action_str}) and {target_vid_str} "
                    f"(predicted action: {target_action_for_prediction}). "
                    f"Minimum predicted distance: {min_distance:.2f}m, which is greater than safety threshold {SAFE_COLLISION_DISTANCE}m. "
                    f"Action {ego_action_str} is SAFE with {target_vid_str}.")


class LLMAgent:
    def __init__(self, api_key: str, env: Any, model_name: str = "deepseek-coder", temperature: float = 0.7):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name
        self.temperature = temperature
        self.env = env 
        self.scenario = Scenario(self.env) # LLMAgent owns the Scenario instance

        # Tools now receive scenario_instance directly
        self.tools = {
            getAvailableActions(self.scenario), 
            isActionSafe(), 
            getAvailableLanes(self.scenario), 
            Get_All_Nearby_Vehicles_Info(self.scenario), 
            CheckTrajectoryConflict(self.scenario), 
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
        - The highway has **two main lanes** (numbered 0 = leftmost, 1 = rightmost) and **one ramp lane** (numbered {RAMP_LANE_IDX} = ramp).
        - The merging area is defined as segment ("c", "d"). The main lane involved is **lane {MERGE_MAIN_LANE_IDX}**, and the ramp lane is **lane {MERGE_RAMP_LANE_IDX}**. Specifically, merging occurs between ("c", "d", {MERGE_MAIN_LANE_IDX}) and ("c", "d", {MERGE_RAMP_LANE_IDX}).
        - Vehicles can be HDV or CAV. You have full observation for all vehicles but cannot communicate with HDVs.
        - Each vehicle knows whether it is on the ramp or the main road, and whether it is in the merging zone.

        --- Lane Change Rules ---
        - **General:** Main lane {MAIN_LANE_INDICES[0]} cannot LANE_LEFT; main lane {MAIN_LANE_INDICES[1]} cannot LANE_RIGHT (except into ramp if in merge zone).
        - **Ramp Specific:**
            - **Ramp segments ("a", "b") and ("b", "c")**: Vehicles in these segments of the ramp (lane {RAMP_LANE_IDX}) can **only go straight** and **cannot change lanes**.
            - **Ramp segment ("c", "d")**: Vehicles in this segment of the ramp (lane {RAMP_LANE_IDX}) are in the **merging zone**. They **can perform LANE_LEFT** action to merge into main lane {MERGE_MAIN_LANE_IDX}, if safe. The end of segment ("c", "d") is the terminal point of the ramp merging area.
        - **Main Lane Specific (Merging Zone):** Main road vehicles in the merging area (lane {MERGE_MAIN_LANE_IDX}, segment ("c", "d")) can **consider LANE_LEFT** to assist ramp vehicles. Main road vehicles can never change right directly into the ramp.

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
            * **Crucially, call `Get_All_Nearby_Vehicles_Info(input='ego')` to get a comprehensive overview of surrounding vehicles.** This tool's output will be a JSON array of simplified vehicle dictionaries. Each dictionary contains 'id', 'type' (CAV/HDV), 'lane_id' (e.g., 'lane_0', 'lane_1', 'lane_2'), and 'distance_from_ego' (positive for front, negative for rear).
            * **After analyzing `Get_All_Nearby_Vehicles_Info`'s output:**
                * **Identify all unique relevant vehicle IDs** from the output. These are the vehicles you must check for conflicts.
                * **Determine Ego's Current Status (Ramp/Main, Merging Zone):** Explicitly recognize if you are a Ramp Vehicle (lane {RAMP_LANE_IDX}) or Main Road Vehicle (lane {MAIN_LANE_INDICES[0]}/{MAIN_LANE_INDICES[1]}), and if you are in the merging zone.
                * **Prioritize ego actions to evaluate based on your current status and merging strategy:**
                    * **General Rule on Frequent Lane Changes:** Do NOT change lanes frequently unless explicitly justified by the merging strategy or a critical safety need. This means `IDLE` or `FASTER` (to maintain speed) are often higher priority in non-critical situations.
                    * **Acceleration Rule:** Unless you are the very first vehicle on a main lane (no front vehicle in your lane) AND there is a ramp vehicle in your right-rear side lane that needs to merge, try to avoid accelerating. Focus on maintaining current speed (IDLE). **If `FASTER` is considered, also verify that current distance to any front vehicle in your lane is at least {MIN_FOLLOWING_DISTANCE_FASTER}m.**
                    * **Prioritization by Vehicle Role/Location:**
                        * If you are a **Ramp Vehicle (lane {RAMP_LANE_IDX}) AND in the merging zone (segment "c","d")**: Prioritize `LANE_LEFT` (if available and safe, as per merging strategy). Then consider `IDLE`, `SLOWER`. **You CANNOT accelerate.**
                        * If you are a **Ramp Vehicle (lane {RAMP_LANE_IDX}) AND NOT in the merging zone (e.g., in "a","b" or "b","c" segments)**: You CANNOT change lanes. Your only available actions are `IDLE`, `SLOWER`. **You CANNOT accelerate.** Prioritize `IDLE`.
                        * If you are a **Main Road Vehicle (lane {MERGE_MAIN_LANE_IDX}) AND in the merging zone (segment "c","d")**: Prioritize `IDLE` or `FASTER` (if no front vehicle and safe). **If a ramp vehicle needs to merge from your right, prioritize `LANE_LEFT` (if safe and available) to yield.** Then `SLOWER`.
                        * If you are a **Main Road Vehicle (lane {MAIN_LANE_INDICES[0]}) or not in the merging zone**: Prioritize `IDLE` or `FASTER` to maintain efficiency. Then `LANE_LEFT`, `LANE_RIGHT`, `SLOWER`.
                * **For each prioritized ego action:**
                    * For *every relevant vehicle* identified, call `Check_Trajectory_Conflict(input='ego_action,target_vehicle_id')`.
                    * **IMPORTANT:** If `Check_Trajectory_Conflict` returns a message starting with `TARGET_VEHICLE_NOT_FOUND`, it means the target vehicle has disappeared from the environment (e.g., left the road, crashed, despawned). This specific check is then **complete for that vehicle, and this result does NOT make the current ego action unsafe.** You should proceed to check other vehicles or confirm the action is safe with respect to *all other* valid vehicles.
                    * **Safety is paramount.** If `Check_Trajectory_Conflict` indicates a `TRAJECTORY CONFLICT_DETECTED` or `LANE_CHANGE_PRECHECK_UNSAFE` or `FOLLOWING_PRECHECK_UNSAFE` for *any* vehicle, that 'ego_action' is immediately deemed unsafe and must NOT be chosen. Discard it and move to the next prioritized action.
                    * **Crucial:** If an action is found to be safe with *all* relevant vehicles (and no `TRAJECTORY CONFLICT_DETECTED`, `LANE_CHANGE_PRECHECK_UNSAFE`, or `FOLLOWING_PRECHECK_UNSAFE` messages were returned), you have found the optimal safe action. **You MUST immediately select this action and output your final decision, without checking any further lower-priority actions.**
            * **HDV Behavior Prediction:** For HDVs, assume they will follow basic traffic rules and driving habits, typically maintaining their current lane and speed (IDLE) unless a specific action is predicted for them.
        3.  **Optimal Strategy Selection:** From the remaining safe actions, select the one that best meets the efficiency and comfort objectives.

        --- Merging Strategy (Crucial for Efficiency and Safety in Merging Zone) ---

        **Your overarching goal is to achieve successful and safe merging for all ramp vehicles while preventing collisions.**

        **General Rule on Frequent Lane Changes:** Do NOT change lanes frequently unless explicitly justified by the merging strategy or a critical safety need.

        **Ramp vehicles (lane {RAMP_LANE_IDX}):**
        - **If in ramp segments ("a", "b") or ("b", "c")**: You CANNOT change lanes. You must continue straight (IDLE or adjust speed) to reach the merging zone. **You CANNOT accelerate.** **This is a strict constraint.**
        - **If in merging zone (ramp lane {RAMP_LANE_IDX}, segment "c", "d")**:
            - If it is safe to merge left into main lane {MERGE_MAIN_LANE_IDX} (as confirmed by tools), **PRIORITIZE LANE_LEFT**. This is your primary objective for successful merging. This is an exception to the frequent lane change rule.
            - Else, adjust speed (IDLE or SLOWER) to seek or create gaps, actively looking for a safe opportunity to merge. **You CANNOT accelerate; instead, consider slowing down to wait for gaps.**
            - If at the very end of the ramp (end of segment "c", "d") with no safe gap to merge, choose SLOWER to avoid collision with the wall or main road vehicles.

        **Main road vehicles (lane {MAIN_LANE_INDICES[0]}-{MAIN_LANE_INDICES[1]}):**
        - **If in merging zone (main lane {MERGE_MAIN_LANE_IDX}, segment "c", "d")**:
            - If a ramp vehicle (from lane {RAMP_LANE_IDX}) needs to merge, and your left lane (lane {MAIN_LANE_INDICES[0]}) is safe and available, **CONSIDER LANE_LEFT to assist** in facilitating traffic flow for the merging vehicle. This is an exception to the frequent lane change rule.
            - Otherwise, adjust speed (FASTER/IDLE/SLOWER) to create a reasonable gap for the ramp vehicle to merge, but avoid heavy braking that disrupts main road flow. **Prioritize maintaining current speed (IDLE) to ensure smooth flow, unless acceleration is clearly beneficial and safe.**
        - In non-merging areas (any lane not in "c", "d" segment): drive efficiently but be prepared to facilitate merging if applicable. **Prioritize maintaining current speed (IDLE) to ensure smooth flow, unless acceleration is clearly beneficial and safe.**

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

    def get_decision(self, observation: Dict[str, Any], log_file=None) -> Dict[str, Any]:
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
                
                if response_message.tool_calls:
                    print(f"LLM Iter {i+1}: Calling Tools: {[tc.function.name for tc in response_message.tool_calls]}")
                    log_file.write(f"LLM Iter {i+1}: Calling Tools: {[tc.function.name for tc in response_message.tool_calls]}\n")
                else:
                    print(f"LLM Iter {i+1}: Text/Final: {response_message.content[:80]}...")
                    log_file.write(f"LLM Iter {i+1}: Text/Final: {response_message.content}\n") 
                
                messages.append(response_message)
                
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args_str = tool_call.function.arguments 
                        
                        tool_to_call = self.tools_map.get(function_name)
                        if tool_to_call:
                            try:
                                parsed_args = json.loads(function_args_str)
                                tool_input = parsed_args.get('input') 
                                
                                tool_output_content = tool_to_call.inference(tool_input)
                                
                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_output_content,
                                })
                                brief_output = tool_output_content.replace('\n', ' ').strip()
                                if len(brief_output) > 100: brief_output = brief_output[:97] + "..."
                                print(f"  -> Tool Output: {function_name} (input: {tool_input}): {brief_output}")
                                log_file.write(f"  -> Tool Output: {function_name} (input: {tool_input}): {tool_output_content}\n") 

                            except json.JSONDecodeError:
                                error_msg = f"Invalid JSON args for {function_name}."
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(f"  -> Tool Error: {function_name} (input: {tool_input}): {error_msg}")
                                log_file.write(f"  -> Tool Error: {function_name} (input: {tool_input}): {error_msg}\n")
                            except KeyError:
                                error_msg = f"Missing 'input' key for {function_name}."
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(f"  -> Tool Error: {function_name} (input: {function_args_str}): {error_msg}")
                                log_file.write(f"  -> Tool Error: {function_name} (input: {function_args_str}): {error_msg}\n")
                            except Exception as e:
                                error_msg = f"Error executing tool {function_name} with args {function_args_str}: {str(e)[:50]}..."
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(f"  -> Tool Error: {function_name} (input: {function_args_str}): {error_msg}")
                                log_file.write(f"  -> Tool Error: {function_name} (input: {function_args_str}): {error_msg}\n")
                        else:
                            error_msg = f"Error: Tool '{function_name}' not found in tools_map."
                            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                            print(f"  -> Tool Error: {function_name}: {error_msg}")
                            log_file.write(f"  -> Tool Error: {function_name}: {error_msg}\n")
                else:
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
                            log_file.write(f"LLM Final Decision: {json.dumps(decision, indent=2)}\n") 
                            return decision 
                        else:
                            print(f"LLM output is JSON but missing required fields. Raw: {llm_output_raw[:80]}...")
                            log_file.write(f"LLM output is JSON but missing required fields. Raw: {llm_output_raw}\n") 
                            messages.append({"role": "assistant", "content": llm_output_raw})
                            continue
                    except json.JSONDecodeError:
                        print(f"LLM did not return valid JSON. Raw: {llm_output_raw[:80]}... Appending as text and continuing.")
                        log_file.write(f"LLM did not return valid JSON. Raw: {llm_output_raw}\n") 
                        messages.append({"role": "assistant", "content": llm_output_raw})
                        continue

            except Exception as e:
                print(f"Error during LLM interaction (outer loop): {e}")
                log_file.write(f"Error during LLM interaction (outer loop): {e}\n")
                return self._fallback_decision(observation, log_file,reason=f"An unexpected error occurred during LLM process: {e}")

        print(f"Max iterations ({max_iterations}) reached without LLM providing a valid JSON decision.")
        log_file.write(f"Max iterations ({max_iterations}) reached without LLM providing a valid JSON decision.\n")
        return self._fallback_decision(observation, log_file,reason=f"Max tool calls ({max_iterations}) reached, no valid JSON decision.")

    def _fallback_decision(self, observation: Dict[str, Any], log_file,reason: str = "LLM interaction failed, falling back to emergency deceleration.") -> Dict[str, Any]:
        print(reason)
        log_file.write(f"FALLBACK DECISION: {reason}\n") 
        return {
            "decision": "SLOWER",
            "reasoning": reason
        }