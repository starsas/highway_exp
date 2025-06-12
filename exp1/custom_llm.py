import openai
import json
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# ==============================================================================
# 1. LLM Tools related definitions and classes
# ==============================================================================

class MockVehicle:
    """
    A mock vehicle class to adapt highway_env.Vehicle to Scenario's expected format.
    This helps the LLM tools to interpret vehicle data consistently.
    """
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self.id = getattr(vehicle, 'id', f"vehicle_{id(vehicle)}") # Use actual ID or generated ID
        self.speed = vehicle.speed
        # lane_id directly uses the numerical lane index
        self.lane_reg= vehicle.lane_index[:2] if vehicle.lane_index else -1
        self.lane_id = vehicle.lane_index[2] if vehicle.lane_index else -1 # -1 for unknown lane
        # lanePosition assumes position[0] is the longitudinal position
        self.lanePosition = vehicle.position[0]
        # Store original lane_index for the merging area check
        self.original_lane_index = vehicle.lane_index
        # Added: Vehicle type (CAV/HDV)
        # Assuming controlled vehicles are CAVs, others are HDVs by default for this context
        self.is_controlled = hasattr(vehicle, 'is_controlled') and vehicle.is_controlled # Flag from highway_env.vehicle.controller.ControlledVehicle


class Scenario:
    """
    Adapts highway_env.Env to the format expected by the LLM tools.
    This class wraps the environment state, providing a simplified view for LLM tool access.
    """
    def __init__(self, env: Any):
        self.env = env
        self.vehicles: Dict[str, MockVehicle] = {}
        self.lanes: Dict[int, Any] = {} # Changed to int keys for lane IDs
        self._update_vehicles()
        self._update_lanes()

    def _update_vehicles(self):
        """Populate the vehicles dictionary with MockVehicle instances."""
        self.vehicles = {}
        
        # Add controlled vehicles first, setting their ID to "ego" if they are the primary one
        if self.env.controlled_vehicles:
            for i, v in enumerate(self.env.controlled_vehicles):
                # We'll refer to the first controlled vehicle as "ego" for LLM
                veh_id = "ego" if i == 0 else getattr(v, 'id', f"vehicle_{id(v)}")
                mock_v = MockVehicle(v)
                mock_v.is_controlled = True # Mark as controlled/CAV
                self.vehicles[veh_id] = mock_v
                v.id = veh_id # Update actual vehicle object's ID for consistency
        
        # Add other (non-controlled) vehicles
        for v in self.env.road.vehicles:
            veh_id = getattr(v, 'id', f"vehicle_{id(v)}")
            if veh_id not in self.vehicles: # Avoid re-adding controlled vehicles
                mock_v = MockVehicle(v)
                mock_v.is_controlled = False # Mark as uncontrolled/HDV
                self.vehicles[veh_id] = mock_v


    def _update_lanes(self):
        """
        Populate the lanes dictionary with simplified lane information.
        Maps numerical lane IDs (e.g., 0, 1, 2, 3) to objects with a `laneIdx` attribute.
        """
        # Assuming lane IDs are from 0 to 3 (3 main lanes + 1 ramp)
        self.lanes = {
            0: type('obj', (object,), {'laneIdx': 0})(),
            1: type('obj', (object,), {'laneIdx': 1})(),
            2: type('obj', (object,), {'laneIdx': 2})(),
            3: type('obj', (object,), {'laneIdx': 3})() # The merge lane
        }

# Helper function to check if a vehicle is in the merging area
def is_in_merging_area(vehicle):
    """Checks if a vehicle is in the merging area."""
    # Based on merge_env_v1.py and demo2_2.py, lane 2 is main, lane 3 is ramp merge
    # The merge section is from node "c" to "d"
    return vehicle.original_lane_index in [("c", "d", 2), ("c", "d", 3)]


def prompts(name, description):
    """
    Decorator to attach name and description to tool *classes* for LLM.
    The decorated class must have an 'inference' method.
    """
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

# --- Tool Classes ---

@prompts(name='Get_Available_Actions', # Changed: spaces replaced with underscores
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
class getAvailableActions:
    """
    Tool to inform LLM about available actions.
    In highway-env DiscreteMetaAction, all 5 actions are generally always available.
    """
    def __init__(self, env: Any) -> None:
        self.env = env # Keep env reference if needed for future dynamic action space

    def inference(self, input: str) -> str:
        availableActions = list(ACTIONS_ALL.keys()) # All actions are generally available
        
        outputPrefix = 'You can ONLY use one of the following actions: \n'
        for action_idx in availableActions:
            outputPrefix += f"{ACTIONS_ALL[action_idx]}--{ACTIONS_DESCRIPTION[action_idx]}; \n"
        
        # Specific advice for LLM based on action types
        if 1 in availableActions:
            outputPrefix += 'You should check idle action as FIRST priority. '
        if 0 in availableActions or 2 in availableActions:
            outputPrefix += 'For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. '
        if 3 in availableActions:
            outputPrefix += 'Consider acceleration action carefully. '
        if 4 in availableActions:
            outputPrefix += 'The deceleration action is LAST priority. '
        
        outputPrefix += """\nTo check decision safety you should follow steps:
        Step 1: Get the vehicles in this lane that you may affect. Acceleration, deceleration and idle action affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Remember to use the proper tools mentioned in the tool list ONCE a time.
        """
        return outputPrefix


@prompts(name='Decision-making_Instructions', # Changed
             description="""This tool gives you a brief introduction about how to ensure that the action you make is safe. The input to this tool should be a string, which is ONLY the action name.""")
class isActionSafe:
    """
    Tool to provide general instructions on how to check action safety.
    This tool doesn't check safety itself but guides the LLM on the process.
    """
    def __init__(self) -> None:
        pass

    def inference(self, action: str) -> str:
        return f"""To check action safety you should follow three steps:
        Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2:(Optional) Get the vehicles in this lane that you may affect, ONLY when you don't know.
        Step 3: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Follow the instructions and remember to use the proper tools mentioned in the tool list once a time.
        """


@prompts(name='Get_Available_Lanes', # Changed
             description="""useful when you want to know the available lanes of the vehicles. like: I want to know the available lanes of the vehicle `ego`. The input to this tool should be a string, representing the id of the vehicle.""")
class getAvailableLanes:
    """
    Tool to determine which lanes are available (current, left, right) for a vehicle.
    Specifically adapted for 3 main lanes (0,1,2) and 1 merge lane (3), with merging area constraint.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env

    def inference(self, vid: str) -> str:
        ego_vehicle_obj = None
        for v in self.env.road.vehicles:
            if getattr(v, 'id', f"vehicle_{id(v)}") == vid:
                ego_vehicle_obj = v
                break
        
        if ego_vehicle_obj is None:
            return f"Vehicle with ID '{vid}' not found."
        
        current_lane_idx = ego_vehicle_obj.lane_index[2]
        
        available_lanes_info = [f"`lane_{current_lane_idx}` is the current lane."]

        # Check for left lane availability
        if current_lane_idx > 0: # Cannot go left from lane 0
            if current_lane_idx == 3: # If on merge lane (index 3), can only go left to main lane 2
                # Only allow lane change if in merging area
                if is_in_merging_area(MockVehicle(ego_vehicle_obj)): # Pass MockVehicle for consistency with is_in_merging_area
                    left_lane_id = 2 # Numerical ID
                    available_lanes_info.append(f"`lane_{left_lane_id}` is to the left of the current lane (merge lane), permitted due to being in merging area.")
                else:
                    available_lanes_info.append(f"Cannot change to `lane_2` from current lane `lane_3` as vehicle is not in merging area.")
            else: # For main lanes, normal left change
                left_lane_id = current_lane_idx - 1
                available_lanes_info.append(f"`lane_{left_lane_id}` is to the left of the current lane.")
        
        # Check for right lane availability
        if current_lane_idx < 3: # From lane 0, 1, or 2, a right lane might exist
            if current_lane_idx == 2: # From lane 2, can go to merge lane 3
                # Only allow lane change if in merging area (main road vehicle might yield/move)
                if is_in_merging_area(MockVehicle(ego_vehicle_obj)): # Pass MockVehicle
                    right_lane_id = 3
                    available_lanes_info.append(f"`lane_{right_lane_id}` is to the right of the current lane (merge lane), permitted due to being in merging area.")
                else:
                    available_lanes_info.append(f"Cannot change to `lane_3` from current lane `lane_2` as vehicle is not in merging area.")
            elif current_lane_idx < 2: # From lane 0 or 1, go to lane 1 or 2
                right_lane_id = current_lane_idx + 1
                available_lanes_info.append(f"`lane_{right_lane_id}` is to the right of the current lane.")
            
        return f"The available lanes of `{vid}` are: " + " ".join(available_lanes_info)


@prompts(name='Get_Lane_Involved_Car', # Changed
             description="""useful when want to know the cars may affect your action in the certain lane. Make sure you have use tool `Get Available Lanes` first. The input is a string, representing the id of the specific lane you want to drive on, DONNOT input multiple lane_id once.""")
class getLaneInvolvedCar:
    """
    Tool to find leading and rearing vehicles in a specified lane relative to ego.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env

    def inference(self, laneID_str: str) -> str: # laneID_str will be 'lane_X'
        try:
            lane_idx = int(laneID_str.split('_')[1])
        except (IndexError, ValueError):
            return "Not a valid lane id format (expected 'lane_X')! Make sure you have use tool `Get Available Lanes` first."
        
        # Find ego vehicle
        ego_vehicle_obj = None
        for v in self.env.road.vehicles:
            if getattr(v, 'id', f"vehicle_{id(v)}") == "ego":
                ego_vehicle_obj = v
                break
        
        if ego_vehicle_obj == None:
            return "Ego vehicle not found."

        leadingCar = None
        rearingCar = None
        min_front_distance = float('inf')
        min_rear_distance = float('inf')

        output_vehicles_info = []

        for v in self.env.road.vehicles:
            if v.id != ego_vehicle_obj.id and v.lane_index[2] == lane_idx: # Check if in the target lane
                distance = v.position[0] - ego_vehicle_obj.position[0]

                vehicle_type = "CAV" if hasattr(v, 'is_controlled') and v.is_controlled else "HDV"
                vehicle_info = f"Vehicle {v.id} (Type: {vehicle_type}, Speed: {round(v.speed, 1)}m/s)"
                
                if distance > 0: # This car is ahead
                    if distance < min_front_distance:
                        min_front_distance = distance
                        leadingCar = v
                        output_vehicles_info.append(f"{vehicle_info} is driving in front of ego car for {round(distance, 2)} meters.")
                else: # This car is behind
                    if abs(distance) < min_rear_distance:
                        min_rear_distance = abs(distance)
                        rearingCar = v
                        output_vehicles_info.append(f"{vehicle_info} is driving behind ego car for {round(abs(distance), 2)} meters.")
        
        output_str = f"On `{laneID_str}`: "
        if not leadingCar and not rearingCar:
            output_str += "There are no cars driving on this lane. This lane is safe, you do not need to check for any vehicle for safety! You can drive on this lane as fast as you can."
        else:
            output_str += "The following vehicles are on this lane: " + " ".join(output_vehicles_info) + ". You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        
        return output_str


@prompts(name='Is_Change_Lane_Conflict_With_Car', # Changed
             description="""useful when you want to know whether change lane to a specific lane is confict with a specific car, ONLY when your decision is change_lane_left or change_lane_right. The input to this tool should be a string of a comma separated string of two, representing the id of the lane you want to change to and the id of the car you want to check.""")
class isChangeLaneConflictWithCar:
    """
    Tool to check if a lane change action conflicts with a specific car.
    Uses a simplified time headway model.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 3.0 # [s] Desired time headway
        self.VEHICLE_LENGTH = 5.0 # [m] Assumed vehicle length

    def inference(self, inputs: str) -> str:
        try:
            laneID_str, vid = inputs.replace(' ', '').split(',')
            lane_idx = int(laneID_str.split('_')[1])
        except (ValueError, IndexError):
            return "Invalid input format. Please provide 'lane_id, vehicle_id'."

        ego_vehicle_obj = None
        target_vehicle_obj = None

        for v in self.env.road.vehicles:
            if getattr(v, 'id', f"vehicle_{id(v)}") == "ego":
                ego_vehicle_obj = v
            if getattr(v, 'id', f"vehicle_{id(v)}") == vid:
                target_vehicle_obj = v
            if ego_vehicle_obj and target_vehicle_obj:
                break
        
        if ego_vehicle_obj == None or target_vehicle_obj == None:
            return "One or both vehicles not found."
        
        # Check merge area constraint for ramp lane changes
        if (ego_vehicle_obj.lane_index[2] == 3 and lane_idx == 2) or \
           (ego_vehicle_obj.lane_index[2] == 2 and lane_idx == 3):
            if not is_in_merging_area(MockVehicle(ego_vehicle_obj)): # Pass MockVehicle
                return f"Change lane from `lane_{ego_vehicle_obj.lane_index[2]}` to `lane_{lane_idx}` is not permitted as vehicle is not in the merging area."


        # Check if the target vehicle is in the lane we are trying to change to
        if target_vehicle_obj.lane_index[2] != lane_idx:
            return f"Vehicle `{vid}` is not in `lane_{lane_idx}`. Please check the lane ID again."

        # Distance calculation based on longitudinal position
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]

        # Target vehicle is ahead of ego
        if distance > 0:
            relativeSpeed = ego_vehicle_obj.speed - target_vehicle_obj.speed
            # Check if ego can maintain safe distance to the leading vehicle after lane change
            if distance - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Change lane to `lane_{lane_idx}` is safe with `{vid}` (leading)."
            else:
                return f"Change lane to `lane_{lane_idx}` may be conflict with `{vid}` (leading), which is unacceptable. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        # Target vehicle is behind ego
        else:
            relativeSpeed = target_vehicle_obj.speed - ego_vehicle_obj.speed
            # Check if the rearing vehicle can maintain safe distance to ego after ego changes lane
            # Use absolute distance for rear vehicle check
            if abs(distance) - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Change lane to `lane_{lane_idx}` is safe with `{vid}` (rearing)."
            else:
                return f"Change lane to `lane_{lane_idx}` may be conflict with `{vid}` (rearing), which is unacceptable. Distance: {abs(distance):.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."


@prompts(name='Is_Acceleration_Conflict_With_Car', # Changed
             description="""useful when you want to know whether acceleration is safe with a specific car, ONLY when your decision is accelerate. The input to this tool should be a string, representing the id of the car you want to check.""")
class isAccelerationConflictWithCar:
    """
    Tool to check if acceleration conflicts with a specific car in the same lane.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 5.0 # [s] A larger headway for acceleration
        self.VEHICLE_LENGTH = 5.0 # [m]
        self.acceleration = 4.0 # [m/s^2] Assumed acceleration value for safety check

    def inference(self, vid: str) -> str:
        ego_vehicle_obj = None
        target_vehicle_obj = None

        for v in self.env.road.vehicles:
            if getattr(v, 'id', f"vehicle_{id(v)}") == "ego":
                ego_vehicle_obj = v
            if getattr(v, 'id', f"vehicle_{id(v)}") == vid:
                target_vehicle_obj = v
            if ego_vehicle_obj and target_vehicle_obj:
                break
        
        if ego_vehicle_obj == None or target_vehicle_obj == None:
            return "One or both vehicles not found."
        
        if target_vehicle_obj.id == ego_vehicle_obj.id:
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        
        if target_vehicle_obj.lane_index[2] != ego_vehicle_obj.lane_index[2]:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]

        if distance > 0: # Check leading car
            relativeSpeed = (ego_vehicle_obj.speed + self.acceleration) - target_vehicle_obj.speed 
            # Simplified look-ahead for acceleration check. Consider braking distance of ego, etc.
            # For simplicity, let's use a doubled vehicle length as buffer in addition to headway
            if distance - (self.VEHICLE_LENGTH * 2) > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Acceleration is safe with `{vid}`."
            else:
                return f"Acceleration may be conflict with `{vid}`, which is unacceptable. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        else: # Other vehicle is behind ego, acceleration is generally safe with rear vehicles
            return f"Acceleration is safe with {vid} (rearing)."


@prompts(name='Is_Keep_Speed_Conflict_With_Car', # Changed
             description="""useful when you want to know whether keep speed is safe with a specific car, ONLY when your decision is keep_speed. The input to this tool should be a string, representing the id of the car you want to check.""")
class isKeepSpeedConflictWithCar:
    """
    Tool to check if maintaining current speed conflicts with a specific car in the same lane.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 5.0 # [s]
        self.VEHICLE_LENGTH = 5.0 # [m]

    def inference(self, vid: str) -> str:
        ego_vehicle_obj = None
        target_vehicle_obj = None

        for v in self.env.road.vehicles:
            if getattr(v, 'id', f"vehicle_{id(v)}") == "ego":
                ego_vehicle_obj = v
            if getattr(v, 'id', f"vehicle_{id(v)}") == vid:
                target_vehicle_obj = v
            if ego_vehicle_obj and target_vehicle_obj:
                break
        
        if ego_vehicle_obj == None or target_vehicle_obj == None:
            return "One or both vehicles not found."
        
        if target_vehicle_obj.id == ego_vehicle_obj.id:
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        
        if target_vehicle_obj.lane_index[2] != ego_vehicle_obj.lane_index[2]:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]
        
        if distance > 0: # Check leading car
            relativeSpeed = ego_vehicle_obj.speed - target_vehicle_obj.speed
            if distance - (self.VEHICLE_LENGTH * 2) > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Keep lane with current speed is safe with {vid}"
            else:
                return f"Keep lane with current speed may be conflict with {vid}, you need to consider decelerate. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        else:
            return f"Keep lane with current speed is safe with {vid} (rearing)."


@prompts(name='Is_Deceleration_Safe', # Changed
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate.The input to this tool should be a string, representing the id of the car you want to check.""")
class isDecelerationSafe:
    """
    Tool to check if deceleration is safe relative to a specific car in the same lane.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 3.0 # [s]
        self.VEHICLE_LENGTH = 5.0 # [m]
        self.deceleration = 3.0 # [m/s^2] Assumed deceleration value

    def inference(self, vid: str) -> str:
        ego_vehicle_obj = None
        target_vehicle_obj = None

        for v in self.env.road.vehicles:
            if getattr(v, 'id', f"vehicle_{id(v)}") == "ego":
                ego_vehicle_obj = v
            if getattr(v, 'id', f"vehicle_{id(v)}") == vid:
                target_vehicle_obj = v
            if ego_vehicle_obj and target_vehicle_obj:
                break
        
        if ego_vehicle_obj == None or target_vehicle_obj == None:
            return "One or both vehicles not found."
        
        if target_vehicle_obj.id == ego_vehicle_obj.id:
            return "You are checking the deceleration of ego car, which is meaningless, input a valid vehicle id please!"
        
        if target_vehicle_obj.lane_index[2] != ego_vehicle_obj.lane_index[2]:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]

        if distance > 0: # Check leading car
            # Assume ego decelerates, so its speed will be lower
            relativeSpeed = (ego_vehicle_obj.speed - self.deceleration) - target_vehicle_obj.speed
            # Check if ego can maintain safe distance to the leading vehicle after deceleration
            if distance - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Deceleration with current speed is safe with {vid}"
            else:
                return f"Deceleration with current speed may be conflict with {vid}, if you have no other choice, slow down as much as possible. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        else:
            return f"Deceleration with current speed is safe with {vid} (rearing)."

# --- 2. LLM Agent Class ---

class LLMAgent:
    def __init__(self, api_key: str, env: Any, model_name: str = "deepseek-coder", temperature: float = 0.7):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com") # DeepSeek API base URL
        self.model_name = model_name
        self.temperature = temperature
        self.env = env # Store env to pass to Scenario and tools
        self.scenario = Scenario(self.env) # Initialize Scenario adapter

        # Initialize tools
        # IMPORTANT: Apply @prompts decorator to the CLASS, not the method.
        # This allows accessing .name and .description directly from the class instance.
        self.tools = {
            getAvailableActions(self.env), # Instance
            isActionSafe(), # Instance
            getAvailableLanes(self.env), # Instance
            getLaneInvolvedCar(self.env), # Instance
            isChangeLaneConflictWithCar(self.env), # Instance
            isAccelerationConflictWithCar(self.env), # Instance
            isKeepSpeedConflictWithCar(self.env), # Instance
            isDecelerationSafe(self.env) # Instance
        }
        # Now tool.name and tool.description are directly on the instance due to the class decorator
        self.tools_map = {tool.name: tool for tool in self.tools}

        # Construct a more detailed system prompt that instructs the LLM on tool use
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        tool_descriptions = ""
        # Iterate over self.tools (which is now a set of instances)
        for tool_obj in self.tools:
            tool_descriptions += f"""
            Tool Name: {tool_obj.name}
            Description: {tool_obj.description}
            """
        prompt = f"""
        You are an autonomous driving agent making decisions in a highway merging area. Your task is to provide the optimal merging strategy for a Connected and Automated Vehicle (CAV, self-vehicle, ID is 'ego').

        **Scenario Description:**
        We are in a highway entrance area with **three main lanes** (numbered 0, 1, 2, from left to right) and **one ramp lane** (numbered 3, merging from the right side of main lane 2).
        The traffic flow is **mixed-type**, consisting of Human-Driven Vehicles (HDVs) and Connected and Automated Vehicles (CAVs).
        - **CAV Information Sharing:** As a CAV, you can obtain precise state information (position, speed, lane, etc.) for all connected vehicles.
        - **HDV Observation Only:** You can only observe the state information of Human-Driven Vehicles (HDVs) via sensors. You cannot directly communicate with them or obtain their future intentions. You need to predict their possible actions based on their current behavior.

        **Your Decision Objective:**
        While ensuring absolute safety, **maximize traffic efficiency in the merging area and the ego vehicle's own throughput efficiency**. This means you need to:
        1. Ensure vehicles maintain reasonable speeds during the merging process, avoiding unnecessary stops or excessively slow driving.
        2. Prioritize smoothly integrating into the main lanes, reducing impact on main lane traffic flow.
        3. Minimize waiting time on the ramp as much as possible.
        4. Consider driving comfort, avoiding sudden accelerations and sudden decelerations.

        **Special Rules:**
        * Lane changes between main lanes and the ramp (lane 2 and lane 3) are only permitted when the vehicle is in the merging area (i.e., main lane 2 or ramp lane 3 near the merge point). Lane changes between other lanes are not subject to this restriction.
        
        **Your Action Space (obtained using the Get_Available_Actions tool):**
        - LANE_LEFT (0): Change lane to the left.
        - IDLE (1): Remain in the current lane with current speed.
        - LANE_RIGHT (2): Change lane to the right.
        - FASTER (3): Accelerate the vehicle.
        - SLOWER (4): Decelerate the vehicle.
        
        **Available Tools:**
        The following are the tools you can call. I will provide you with the output of these tools. You must acknowledge the information and be ready for the next step, or respond with a final JSON decision when explicitly asked.
        {tool_descriptions}
        
        **Decision-Making Process Guide:**
        Your decision-making process will proceed in distinct phases. I will prompt you at each phase with the results of tool calls. You must acknowledge the information and be ready for the next step, or respond with a final JSON decision when explicitly asked.

        **Phase 1: Initial Understanding & Action Space Query**
        1.  **Perception and Understanding:** Carefully analyze the current traffic scene information, especially the ego vehicle's position (whether in the merging area) and the speed and distance of surrounding vehicles (distinguishing between CAVs and HDVs).
            * **HDV Behavior Prediction:** For HDVs, assume they will follow basic traffic rules and driving habits. In the absence of clear intentions, tend to drive conservatively, e.g., maintaining current lane and speed, unless there's an obstacle ahead.
        2.  **Get Available Actions:** I will provide you with the output of `Get_Available_Actions` tool. Acknowledge this.
        
        **Phase 2: Safety Assessment**
        3.  **Safety Assessment (Core):** For each potential action, I will provide you with information about involved vehicles or safety checks. You must acknowledge this and update your understanding of safe actions.
            * **Safety is paramount. If an action is determined to be unsafe, it must not be chosen.**
            * **Conservative Strategy for HDVs:** When interacting with HDVs, adopt a more conservative strategy, allowing for larger safety margins.
        
        **Phase 3: Strategy Selection & Final Decision**
        4.  **Strategy Selection (Based on Efficiency and Comfort):** After all relevant safety checks are complete, you will be explicitly asked to make a final decision. Among all safe actions, select the optimal action based on your optimization objectives.
            * **Ramp Vehicle (CAV):** In the merging area, if a safe lane change to the main lane is possible, prioritize executing it. If an immediate lane change is not possible, try adjusting speed (FASTER/SLOWER/IDLE) to find or create a safe gap, aiming to complete the merge at a faster speed. If unable to change lanes at the end of the ramp, it is better to decelerate and wait.
            * **Main Lane Vehicle (CAV):** In the merging area, if there is a ramp vehicle needing to merge, and the left lane is safe and available, consider changing to the left lane to facilitate overall traffic flow efficiency. If unable to change left, try adjusting speed (SLOWER/FASTER/IDLE) to create a reasonable gap for the ramp vehicle, but avoid a significant reduction in your own speed.
            * **All CAVs:** In non-merging areas, maintain efficient driving, avoiding unnecessary deceleration.
        5.  **Output Instruction:** You must return your final decision in JSON format. Your response must always be JSON and include the `decision` (string) and `reasoning` (string) fields.
            * The `decision` field must be one of `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`.
            * The `reasoning` field should explain in detail why you made this decision, specifically referencing the tool outputs.
        
        **Note:**
        - If no safe action is available after all checks, you must choose the `SLOWER` action as an emergency evasive maneuver and explain why.
        - Your final response MUST be a valid JSON object.
        """
        prompt1 = f"""
        You are an autonomous driving agent making decisions in a highway merging area. Your task is to provide the optimal merging strategy for a Connected and Automated Vehicle (CAV, self-vehicle, ID is 'ego').

        **Scenario Description:**
        We are in a highway entrance area with **three main lanes** (numbered 0, 1, 2, from left to right) and **one ramp lane** (numbered 3, merging from the right side of main lane 2).
        The traffic flow is **mixed-type**, consisting of Human-Driven Vehicles (HDVs) and Connected and Automated Vehicles (CAVs).
        - **CAV Information Sharing:** As a CAV, you can obtain precise state information (position, speed, lane, etc.) for all connected vehicles.
        - **HDV Observation Only:** You can only observe the state information of Human-Driven Vehicles (HDVs) via sensors. You cannot directly communicate with them or obtain their future intentions. You need to predict their possible actions based on their current behavior.

        **Your Decision Objective:**
        While ensuring absolute safety, **maximize traffic efficiency in the merging area and the ego vehicle's own throughput efficiency**. This means you need to:
        1. Ensure vehicles maintain reasonable speeds during the merging process, avoiding unnecessary stops or excessively slow driving.
        2. Prioritize smoothly integrating into the main lanes, reducing impact on main lane traffic flow.
        3. Minimize waiting time on the ramp as much as possible.
        4. Consider driving comfort, avoiding sudden accelerations and decelerations.

        **Special Rules:**
        * Lane changes between main lanes and the ramp (lane 2 and lane 3) are only permitted when the vehicle is in the merging area (i.e., main lane 2 or ramp lane 3 near the merge point). Lane changes between other lanes are not subject to this restriction.
        * Try to keep a safe distance to the car in front of you.
        * DONOT change lane frequently. If you want to change lane, double-check the safety of vehicles on target lane.
        * If your speed and leading car speed is near and distance is
            delete this item: DONOT change lane frequently. If you want to change lane, double-check the safety of vehicles on target lane.
        * **Within the merging area (lane 2 or lane 3):**
            * **Ramp Vehicle (Lane 3):** If there is a safe space to merge into main lane 2, prioritize changing lanes.
            * **Main Lane Vehicle (Lane 2):** If there is a ramp vehicle (from lane 3) needing to merge, and the left lane (lane 1) is safe and available, ignore this item: "DONOT change lane frequently. If you want to change lane, double-check the safety of vehicles on target lane.",and consider yielding by changing to lane 1.

        **Your Action Space (obtained using the Get_Available_Actions tool):**
        - LANE_LEFT (0): Change lane to the left.
        - IDLE (1): Remain in the current lane with current speed.
        - LANE_RIGHT (2): Change lane to the right.
        - FASTER (3): Accelerate the vehicle.
        - SLOWER (4): Decelerate the vehicle.
        
        **Available Tools:**
        The following are the tools you can call. You must strictly follow their names and descriptions and provide correct input parameters.
        {tool_descriptions}
        
        **Decision-Making Process Guide:**
        1.  **Perception and Understanding:** Carefully analyze the provided current traffic scene information, especially the ego vehicle's position (whether in the merging area) and the speed and distance of surrounding vehicles (distinguishing between CAVs and HDVs).
            * **HDV Behavior Prediction:** For HDVs, assume they will follow basic traffic rules and driving habits. In the absence of clear intentions, tend to drive conservatively, e.g., maintaining current lane and speed, unless there's an obstacle ahead.
        2.  **Get Available Actions:** Prioritize using the `Get_Available_Actions` tool to understand all available actions in the current situation.
        3.  **Safety Assessment (Core):**
            * For each possible action (especially lane change and acceleration/deceleration actions), you must use the `Decision-making_Instructions` tool to understand how to check its safety.
            * Then, use the `Get_Lane_Involved_Car` tool to get all relevant vehicles in the affected lane.
            * For each relevant vehicle and specific action, call the corresponding safety check tool (e.g., `Is_Change_Lane_Conflict_With_Car`, `Is_Acceleration_Conflict_With_Car`, etc.) to assess potential conflicts.
            * **Safety is paramount. If an action is determined to be unsafe, it must not be chosen.**
            * **Conservative Strategy for HDVs:** When interacting with HDVs, adopt a more conservative strategy, allowing for larger safety margins.
        4.  **Strategy Selection (Based on Efficiency and Comfort):** Among all safe actions, select the optimal action based on your optimization objectives.
            * **Ramp Vehicle (CAV):** In the merging area, if a safe lane change to the main lane is possible, prioritize executing it. If an immediate lane change is not possible, try adjusting speed (FASTER/SLOWER/IDLE) to find or create a safe gap, aiming to complete the merge at a faster speed. If unable to change lanes at the end of the ramp, it is better to decelerate and wait.
            * **Main Lane Vehicle (CAV):** In the merging area, if there is a ramp vehicle needing to merge, and the left lane is safe and available, consider changing to the left lane to facilitate overall traffic flow efficiency. If unable to change left, try adjusting speed (SLOWER/FASTER/IDLE) to create a reasonable gap for the ramp vehicle, but avoid a significant reduction in your own speed.
            * **All CAVs:** In non-merging areas, maintain efficient driving, avoiding unnecessary deceleration.
        5.  **Output Instruction:** Return your final decision in JSON format. Your response must always be JSON and include the `decision` (string) and `reasoning` (string) fields.
            * The `decision` field must be one of `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`.
            * The `reasoning` field should explain in detail why you made this decision, specifically referencing the outputs of the tools used.
        
        **Note:**
        - If no safe action is available, you must choose the `SLOWER` action as an emergency evasive maneuver and explain why.
        - Your output must be a valid JSON object.
        """
        return prompt

   
    def get_decision(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Initial user prompt with full observation
        user_prompt_initial = f"""
        Current traffic scene perception information is as follows:
        Ego Vehicle Info: {json.dumps(observation['ego_vehicle'], indent=2)}
        Nearby Vehicles Info: {json.dumps(observation['nearby_vehicles'], indent=2)}
        Road Info: {json.dumps(observation['road_info'], indent=2)}
        Current Traffic Density: {observation.get('traffic_density', 'unknown')}
        
        Please analyze this initial traffic situation. I will then provide you with results from specific tool calls step-by-step.
        """
        messages.append({"role": "user", "content": user_prompt_initial})

        try:
            # Step 0: LLM processes initial observation and system prompt
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # print(f"LLM Initial Acknowledgement: {response.choices[0].message.content}")
            llm_output = response.choices[0].message.content
            clean_llm_output = llm_output.strip()
            if clean_llm_output.startswith('```json') and clean_llm_output.endswith('```'):
                llm_output = clean_llm_output[len('```json'):-len('```')].strip()
            
            decision = json.loads(llm_output)
            # print(decision)
            if all(k in decision for k in ["decision", "reasoning"]):
                return decision
            else:
                print(f"LLM final output missing required fields: {llm_output}")
                return self._fallback_decision(observation, reason="Invalid final JSON format from LLM.")
            # --- Tool Chain Execution ---
            """
            # Phase 1: Get Available Actions
            tool_name = "Get_Available_Actions"
            tool_input = "ego"
            tool_output = self.tools_map[tool_name].inference(tool_input)
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{tool_name}",
                "name": tool_name,
                "content": tool_output,
            })
            print(f"Tool Call: {tool_name}({tool_input}) -> Output: {tool_output}")
            
            messages.append({"role": "user", "content": f"I have executed the tool `{tool_name}` with input `{tool_input}`. Here is the output: `{tool_output}`. Please confirm you understand the available actions."})
            response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=self.temperature)
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            print(f"LLM after {tool_name}: {response.choices[0].message.content}")

            # Phase 2: Get Decision-making Instructions
            tool_name = "Decision-making_Instructions"
            tool_input = "safety_check_guidance" # Placeholder input to get instructions
            tool_output = self.tools_map[tool_name].inference(tool_input)
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{tool_name}",
                "name": tool_name,
                "content": tool_output,
            })
            print(f"Tool Call: {tool_name}({tool_input}) -> Output: {tool_output}")

            messages.append({"role": "user", "content": f"I have executed the tool `{tool_name}` for general safety instructions. Here is the output: `{tool_output}`. Please acknowledge this guidance."})
            response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=self.temperature)
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            print(f"LLM after {tool_name}: {response.choices[0].message.content}")

            # Phase 3: Get Available Lanes
            tool_name = "Get_Available_Lanes"
            tool_input = "ego"
            tool_output = self.tools_map[tool_name].inference(tool_input)
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{tool_name}",
                "name": tool_name,
                "content": tool_output,
            })
            print(f"Tool Call: {tool_name}({tool_input}) -> Output: {tool_output}")

            messages.append({"role": "user", "content": f"I have executed the tool `{tool_name}` with input `{tool_input}`. Here is the output: `{tool_output}`. Please understand the available lanes."})
            response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=self.temperature)
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            print(f"LLM after {tool_name}: {response.choices[0].message.content}")

            # Phase 4: Safety Checks for ALL possible actions, explicitly.
            # This requires knowing all potential actions and then iterating through them.
            # Let's define a prioritized list of actions to check.
            actions_to_check = ['IDLE', 'LANE_LEFT', 'LANE_RIGHT', 'FASTER', 'SLOWER']
            safe_actions_info = {} # To store results of safety checks for each action

            ego_lane_id = observation['ego_vehicle']['lane_id']
            ego_speed = observation['ego_vehicle']['speed']

            # Iterate through actions and ask LLM to consider safety
            for action_str in actions_to_check:
                # Determine relevant lane for the action
                target_lane_id_for_check = ego_lane_id
                if action_str == 'LANE_LEFT':
                    if ego_lane_id == 3: target_lane_id_for_check = 2
                    elif ego_lane_id > 0: target_lane_id_for_check = ego_lane_id - 1
                    else: continue # Cannot go left from lane 0
                elif action_str == 'LANE_RIGHT':
                    if ego_lane_id == 2: target_lane_id_for_check = 3
                    elif ego_lane_id < 2: target_lane_id_for_check = ego_lane_id + 1
                    else: continue # Cannot go right from lane 3 or if already rightmost
                elif action_str == 'SLOWER' and ego_speed <= 0.5: # Don't try to slow down if already stopped
                    continue

                # Get involved cars for the relevant lane
                tool_name_get_involved_cars = "Get_Lane_Involved_Car"
                tool_input_get_involved_cars = f"lane_{target_lane_id_for_check}"
                tool_output_get_involved_cars = self.tools_map[tool_name_get_involved_cars].inference(tool_input_get_involved_cars)
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{tool_name_get_involved_cars}_{action_str}",
                    "name": tool_name_get_involved_cars,
                    "content": tool_output_get_involved_cars,
                })
                print(f"Tool Call: {tool_name_get_involved_cars}({tool_input_get_involved_cars}) -> Output: {tool_output_get_involved_cars}")

                # Ask LLM to determine safety of *this specific action* with involved cars
                safety_check_tool_name = None
                if action_str == 'LANE_LEFT' or action_str == 'LANE_RIGHT':
                    safety_check_tool_name = "Is_Change_Lane_Conflict_With_Car"
                elif action_str == 'FASTER':
                    safety_check_tool_name = "Is_Acceleration_Conflict_With_Car"
                elif action_str == 'IDLE':
                    safety_check_tool_name = "Is_Keep_Speed_Conflict_With_Car"
                elif action_str == 'SLOWER':
                    safety_check_tool_name = "Is_Deceleration_Safe"
                
                # Iterate through nearby vehicles to check safety (LLM's internal loop)
                # This part is tricky. The LLM would typically decide which cars to check based on 'Get_Lane_Involved_Car' output.
                # Here, we'll give it the output, and ask it to confirm safety based on it.
                # If there are no cars involved, the safety tool should confirm safety.
                
                # A more explicit way to ensure LLM performs checks:
                # Extract car IDs from tool_output_get_involved_cars if possible.
                involved_car_ids = []
                # Simple parsing: look for "Vehicle <ID>" patterns
                for line in tool_output_get_involved_cars.split('\n'):
                    if "Vehicle" in line and "ID:" not in line: # Avoid double counting or misparsing
                        parts = line.split("Vehicle ")[1].split(" ")
                        if len(parts) > 0:
                            involved_car_ids.append(parts[0].strip())
                
                safety_outputs_for_this_action = []
                if not involved_car_ids:
                    # If no cars, it's generally safe (except for explicit lane change rules)
                    safety_outputs_for_this_action.append(f"No vehicles found in lane {target_lane_id_for_check}. {action_str} is likely safe related to other vehicles.")
                else:
                    for car_id in involved_car_ids:
                        if safety_check_tool_name:
                            tool_input_safety_check = f"{f'lane_{target_lane_id_for_check},' if 'Lane' in action_str else ''}{car_id}"
                            try:
                                individual_safety_output = self.tools_map[safety_check_tool_name].inference(tool_input_safety_check)
                                safety_outputs_for_this_action.append(individual_safety_output)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": f"call_{safety_check_tool_name}_{action_str}_{car_id}",
                                    "name": safety_check_tool_name,
                                    "content": individual_safety_output,
                                })
                                print(f"Tool Call: {safety_check_tool_name}({tool_input_safety_check}) -> Output: {individual_safety_output}")
                            except Exception as e:
                                error_msg = f"Error executing safety tool {safety_check_tool_name} for {action_str} with {car_id}: {e}"
                                safety_outputs_for_this_action.append(error_msg)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": f"call_error_{safety_check_tool_name}_{action_str}_{car_id}",
                                    "name": safety_check_tool_name,
                                    "content": error_msg,
                                })
                                print(error_msg)
                        else:
                            safety_outputs_for_this_action.append(f"No specific safety tool for action {action_str} with {car_id}.")

                # Prompt LLM to understand safety checks for this action
                messages.append({"role": "user", "content": f"I have executed safety checks for action `{action_str}` in lane `lane_{target_lane_id_for_check}`. Here are the results:\n" + "\n".join(safety_outputs_for_this_action) + "\nPlease analyze these results and tell me if action `{action_str}` is definitively safe based on these checks, and why."})
                response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=self.temperature)
                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                print(f"LLM safety analysis for {action_str}: {response.choices[0].message.content}")

                # Store LLM's assessment of safety for this action
                safe_actions_info[action_str] = response.choices[0].message.content # LLM's opinion on safety

            """
            # Phase 5: Final Decision Phase
            final_decision_prompt = f"""
            Based on ALL the information I have provided from the tool calls and your subsequent analyses, including available actions, safety instructions, vehicle information on different lanes, and safety checks for each action, please now make your FINAL optimal merging decision.

            You have analyzed the safety of various actions. Now, considering the overall objectives (safety first, then efficiency and comfort), choose the best action.

            Your response MUST be a valid JSON object containing 'decision' (string) and 'reasoning' (string).
            - 'decision' must be one of 'LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER'.
            - 'reasoning' must explain your decision, specifically referencing the tool outputs and your safety analyses.
            
            If no safe action is found based on the tool outputs, you MUST choose 'SLOWER' as an emergency evasive maneuver.
            """
            messages.append({"role": "user", "content": final_decision_prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            llm_output = response.choices[0].message.content
            decision = json.loads(llm_output)

            if all(k in decision for k in ["decision", "reasoning"]):
                return decision
            else:
                print(f"LLM final output missing required fields: {llm_output}")
                return self._fallback_decision(observation, reason="Invalid final JSON format from LLM.")

        except json.JSONDecodeError:
            print(f"LLM did not return valid JSON for final decision")
            return self._fallback_decision(observation, reason="LLM final output not valid JSON.")
        except Exception as e:
            print(f"Error during LLM decision-making process: {e}")
            return self._fallback_decision(observation, reason=f"An unexpected error occurred during LLM process: {e}")

    def _fallback_decision(self, observation: Dict[str, Any], reason: str = "LLM interaction failed, falling back to emergency deceleration.") -> Dict[str, Any]:
        print(reason)
        ego_lane_id_int = observation['ego_vehicle']['lane_id'] if 'ego_vehicle' in observation and observation['ego_vehicle'] else -1 
        return {
            "decision": "SLOWER",
            # "target_lane": ego_lane_id_int, # Fallback lane should be current lane
            "reasoning": reason
        }    

    def get_decision1(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interacts with the LLM to get a driving decision.
        Manages tool calls and parsing of LLM's responses.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        user_prompt = f"""
        Current traffic scene perception information is as follows:
        Ego Vehicle Info: {json.dumps(observation['ego_vehicle'], indent=2)}
        Nearby Vehicles Info: {json.dumps(observation['nearby_vehicles'], indent=2)}
        Road Info: {json.dumps(observation['road_info'], indent=2)}
        Current Traffic Density: {observation.get('traffic_density', 'unknown')}
        
        Please analyze the current traffic situation and use your tools to infer and generate an optimal merging decision.
        """
        messages.append({"role": "user", "content": user_prompt})

        max_tool_calls = 5 # Limit to prevent infinite loops of tool calls
        
        for i in range(max_tool_calls):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=[
                        {"type": "function", "function": {"name": tool_obj.name, "description": tool_obj.description, "parameters": {"type": "object", "properties": {"input": {"type": "string"}}}}}
                        for tool_obj in self.tools # Iterate over instances, not values
                    ],
                    tool_choice="auto", # Let LLM decide whether to call a tool
                    temperature=self.temperature,
                    response_format={"type": "json_object"} if i == max_tool_calls -1 else None
                )
                
                response_message = response.choices[0].message
                print(response_message)
                if response_message.tool_calls:
                    messages.append(response_message)
                    
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = tool_call.function.arguments
                        
                        # Fix: Get tool instance from self.tools_map
                        tool_to_call = self.tools_map.get(function_name) 
                        if tool_to_call:
                            try:
                                parsed_args = json.loads(function_args).get('input', function_args) if isinstance(function_args, str) and function_args.strip().startswith('{') else function_args
                                tool_output_content = tool_to_call.inference(parsed_args)
                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_output_content,
                                })
                            except Exception as e:
                                error_msg = f"Error executing tool {function_name} with args {function_args}: {e}"
                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": error_msg,
                                })
                                print(error_msg)
                        else:
                            error_msg = f"Error: Tool '{function_name}' not found."
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": error_msg,
                            })
                            print(error_msg)
                else:
                    llm_output = response_message.content
                    try:
                        decision = json.loads(llm_output)
                        if all(k in decision for k in ["decision", "reasoning"]):
                            return decision
                        else:
                            print(f"LLM output missing required fields: {llm_output}")
                            raise ValueError("Invalid LLM output format")
                    except json.JSONDecodeError:
                        print(f"LLM did not return valid JSON: {llm_output}. Retrying with force JSON.")
                        messages.append({"role": "assistant", "content": llm_output})
            
            except Exception as e:
                print(f"Error during LLM interaction: {e}")
                break

        print("LLM failed to provide a valid decision after multiple attempts or errors. Falling back to emergency deceleration.")
        # ego_lane_id_int = observation['ego_vehicle']['lane_id'] if 'ego_vehicle' in observation and observation['ego_vehicle'] else -1 

        return {
            "decision": "SLOWER",
            "reasoning": "LLM interaction failed, falling back to emergency deceleration."
        }