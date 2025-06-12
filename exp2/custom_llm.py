import openai
import json
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# ==============================================================================
# 1. LLM Tools 相关的定义和类
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
        self.lane_id = vehicle.lane_index[2] if vehicle.lane_index else -1 # -1 for unknown lane
        # lanePosition assumes position[0] is the longitudinal position
        self.lanePosition = vehicle.position[0]
        # Store original lane_index for the merging area check
        self.original_lane_index = vehicle.lane_index

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
        # Ensure 'ego' vehicle is handled first if it's in controlled_vehicles
        
        # Add controlled vehicles first, setting their ID to "ego" if they are the primary one
        if self.env.controlled_vehicles:
            for i, v in enumerate(self.env.controlled_vehicles):
                # We'll refer to the first controlled vehicle as "ego" for LLM
                veh_id = "ego" if i == 0 else getattr(v, 'id', f"vehicle_{id(v)}")
                self.vehicles[veh_id] = MockVehicle(v)
                v.id = veh_id # Update actual vehicle object's ID for consistency
        
        # Add other (non-controlled) vehicles
        for v in self.env.road.vehicles:
            veh_id = getattr(v, 'id', f"vehicle_{id(v)}")
            if veh_id not in self.vehicles: # Avoid re-adding controlled vehicles
                self.vehicles[veh_id] = MockVehicle(v)


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
    """判断车辆是否在合流区域"""
    # Based on merge_env_v1.py and demo2_2.py, lane 2 is main, lane 3 is ramp merge
    # The merge section is from node "c" to "d"
    return vehicle.original_lane_index in [("c", "d", 2), ("c", "d", 3)]


def prompts(name, description):
    """Decorator to attach name and description to tool functions for LLM."""
    def decorator(func):
        func.name = name
        func.description = description
        return func
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

class getAvailableActions:
    """
    Tool to inform LLM about available actions.
    In highway-env DiscreteMetaAction, all 5 actions are generally always available.
    """
    def __init__(self, env: Any) -> None:
        self.env = env # Keep env reference if needed for future dynamic action space

    @prompts(name='Get Available Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
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


class isActionSafe:
    """
    Tool to provide general instructions on how to check action safety.
    This tool doesn't check safety itself but guides the LLM on the process.
    """
    def __init__(self) -> None:
        pass

    @prompts(name='Decision-making Instructions',
             description="""This tool gives you a brief introduction about how to ensure that the action you make is safe. The input to this tool should be a string, which is ONLY the action name.""")
    def inference(self, action: str) -> str:
        return f"""To check action safety you should follow three steps:
        Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2:(Optional) Get the vehicles in this lane that you may affect, ONLY when you don't know.
        Step 3: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Follow the instructions and remember to use the proper tools mentioned in the tool list once a time.
        """


class getAvailableLanes:
    """
    Tool to determine which lanes are available (current, left, right) for a vehicle.
    Specifically adapted for 3 main lanes (0,1,2) and 1 merge lane (3), with merging area constraint.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env

    @prompts(name='Get Available Lanes',
             description="""useful when you want to know the available lanes of the vehicles. like: I want to know the available lanes of the vehicle `ego`. The input to this tool should be a string, representing the id of the vehicle.""")
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


class getLaneInvolvedCar:
    """
    Tool to find leading and rearing vehicles in a specified lane relative to ego.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env

    @prompts(name='Get Lane Involved Car',
             description="""useful when want to know the cars may affect your action in the certain lane. Make sure you have use tool `Get Available Lanes` first. The input is a string, representing the id of the specific lane you want to drive on, DONNOT input multiple lane_id once.""")
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
        
        if ego_vehicle_obj is None:
            return "Ego vehicle not found."

        leadingCar = None
        rearingCar = None
        min_front_distance = float('inf')
        min_rear_distance = float('inf')

        for v in self.env.road.vehicles:
            if v.id != ego_vehicle_obj.id and v.lane_index[2] == lane_idx: # Check if in the target lane
                # Use vehicle's actual position for distance calculation
                distance = v.position[0] - ego_vehicle_obj.position[0]

                if distance > 0: # This car is ahead
                    if distance < min_front_distance:
                        min_front_distance = distance
                        leadingCar = v
                else: # This car is behind
                    if abs(distance) < min_rear_distance:
                        min_rear_distance = abs(distance)
                        rearingCar = v
        
        output_str = f"On `{laneID_str}`: "
        if not leadingCar and not rearingCar:
            output_str += "There are no cars driving on this lane. This lane is safe, you do not need to check for any vehicle for safety! You can drive on this lane as fast as you can."
        elif leadingCar and not rearingCar:
            distance = round(min_front_distance, 2)
            leading_car_vel = round(leadingCar.speed, 1)
            output_str += f"{leadingCar.id} is driving at {leading_car_vel}m/s on `{laneID_str}`, and it's driving in front of ego car for {distance} meters. You need to make sure that your actions do not conflict with this vehicle."
        elif not leadingCar and rearingCar:
            distance = round(min_rear_distance, 2)
            rearing_car_vel = round(rearingCar.speed, 1)
            output_str += f"{rearingCar.id} is driving at {rearing_car_vel}m/s on `{laneID_str}`, and it's driving behind ego car for {distance} meters. You need to make sure that your actions do not conflict with this vehicle."
        else: # Both leading and rearing cars exist
            distance_leading = round(min_front_distance, 2)
            leading_car_vel = round(leadingCar.speed, 1)
            distance_rearing = round(min_rear_distance, 2)
            rearing_car_vel = round(rearingCar.speed, 1)

            output_str += (f"{leadingCar.id} (speed: {leading_car_vel}m/s, {distance_leading}m in front) and "
                           f"{rearingCar.id} (speed: {rearing_car_vel}m/s, {distance_rearing}m behind) "
                           f"are driving on `{laneID_str}`. You need to make sure that your actions do not conflict with each of the vehicles mentioned.")
        
        return output_str


class isChangeLaneConflictWithCar:
    """
    Tool to check if a lane change action conflicts with a specific car.
    Uses a simplified time headway model.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 3.0 # [s] Desired time headway
        self.VEHICLE_LENGTH = 5.0 # [m] Assumed vehicle length

    @prompts(name='Is Change Lane Conflict With Car',
             description="""useful when you want to know whether change lane to a specific lane is confict with a specific car, ONLY when your decision is change_lane_left or change_lane_right. The input to this tool should be a string of a comma separated string of two, representing the id of the lane you want to change to and the id of the car you want to check.""")
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
        
        if ego_vehicle_obj is None or target_vehicle_obj is None:
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


class isAccelerationConflictWithCar:
    """
    Tool to check if acceleration conflicts with a specific car in the same lane.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 5.0 # [s] A larger headway for acceleration
        self.VEHICLE_LENGTH = 5.0 # [m]
        self.acceleration = 4.0 # [m/s^2] Assumed acceleration value for safety check

    @prompts(name='Is Acceleration Conflict With Car',
             description="""useful when you want to know whether acceleration is safe with a specific car, ONLY when your decision is accelerate. The input to this tool should be a string, representing the id of the car you want to check.""")
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
        
        if ego_vehicle_obj is None or target_vehicle_obj is None:
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


class isKeepSpeedConflictWithCar:
    """
    Tool to check if maintaining current speed conflicts with a specific car in the same lane.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 5.0 # [s]
        self.VEHICLE_LENGTH = 5.0 # [m]

    @prompts(name='Is Keep Speed Conflict With Car',
             description="""useful when you want to know whether keep speed is safe with a specific car, ONLY when your decision is keep_speed. The input to this tool should be a string, representing the id of the car you want to check.""")
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
        
        if ego_vehicle_obj is None or target_vehicle_obj is None:
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


class isDecelerationSafe:
    """
    Tool to check if deceleration is safe relative to a specific car in the same lane.
    """
    def __init__(self, env: Any) -> None: # Changed to receive env directly
        self.env = env
        self.TIME_HEAD_WAY = 3.0 # [s]
        self.VEHICLE_LENGTH = 5.0 # [m]
        self.deceleration = 3.0 # [m/s^2] Assumed deceleration value

    @prompts(name='Is Deceleration Safe',
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate.The input to this tool should be a string, representing the id of the car you want to check.""")
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
        
        if ego_vehicle_obj is None or target_vehicle_obj is None:
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
        self.tools = {
            "Get Available Actions": getAvailableActions(self.env),
            "Decision-making Instructions": isActionSafe(),
            "Get Available Lanes": getAvailableLanes(self.env), # Pass env directly
            "Get Lane Involved Car": getLaneInvolvedCar(self.env), # Pass env directly
            "Is Change Lane Conflict With Car": isChangeLaneConflictWithCar(self.env), # Pass env directly
            "Is Acceleration Conflict With Car": isAccelerationConflictWithCar(self.env), # Pass env directly
            "Is Keep Speed Conflict With Car": isKeepSpeedConflictWithCar(self.env), # Pass env directly
            "Is Deceleration Safe": isDecelerationSafe(self.env) # Pass env directly
        }
        self.tools_map = {tool.name: tool for tool in self.tools.values()}

        # Construct a more detailed system prompt that instructs the LLM on tool use
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        tool_descriptions = ""
        for tool_name, tool_obj in self.tools.items():
            tool_descriptions += f"""
            Tool Name: {tool_obj.name}
            Description: {tool_obj.description}
            """
        
        prompt = f"""
        你是一个在高速公路合流区域进行决策的自动驾驶智能体。你的任务是为一辆网联车辆（自车，ID为'ego'）提供最优的合流策略。
        
        **场景描述：**
        我们正在一个具有**三条主车道**（编号0, 1, 2，从左到右）和**一条匝道**（编号3，从主车道2右侧汇入）的高速公路入口区域。
        交通流是**混合类型**，包含人类驾驶车辆和网联车辆。你需要预测人类车辆的行为。
        
        **你的决策目标：**
        在保证绝对安全的前提下，**最大化合流区域的交通效率和自车自身的通行效率**。这意味着你需要：
        1. 确保车辆在合流过程中保持合理的车速，避免不必要的停车或过慢行驶。
        2. 优先考虑流畅地融入主车道，减少对主车道交通流的影响。
        3. 尽可能缩短在匝道上的等待时间。
        4. 兼顾驾驶舒适性，避免急加速和急减速。

        **特殊规则：**
        * 只有当车辆处于合流区域（即主车道2或匝道3在合流点附近）时，才能进行主车道和匝道之间的变道操作。其他车道之间的变道不受此限制。
        
        **你的动作空间（通过工具Get Available Actions获取）：**
        - LANE_LEFT (0): 向左变道
        - IDLE (1): 保持当前车道和速度
        - LANE_RIGHT (2): 向右变道
        - FASTER (3): 加速
        - SLOWER (4): 减速
        
        **你可用的工具：**
        以下是你能够调用的工具。你必须严格按照工具的名称和描述来使用它们，并提供正确的输入参数。
        {tool_descriptions}
        
        **决策流程指南：**
        1.  **感知与理解：** 仔细分析提供的当前交通场景信息，特别是自车位置（是否在合流区）、周围车辆的速度和距离。
        2.  **获取可用动作：** 优先使用 `Get Available Actions` 工具来了解当前情境下可以执行的所有动作。
        3.  **安全性评估 (核心)：**
            * 对于每一个可能的动作（特别是变道和加减速动作），你必须使用 `Decision-making Instructions` 工具来了解如何检查其安全性。
            * 然后，利用 `Get Lane Involved Car` 工具获取受影响车道上的所有相关车辆。
            * 针对每个相关车辆和具体动作，调用相应的安全检查工具（例如 `Is Change Lane Conflict With Car`, `Is Acceleration Conflict With Car` 等）来评估潜在冲突。
            * **安全是第一位的。如果某个动作被判定为不安全，则不应选择该动作。**
        4.  **策略选择（基于效率和舒适性）：** 在所有安全的动作中，根据你的优化目标，选择最佳动作。
            * **匝道车：** 在合流区，如果能安全变道，优先变道。如果不能，则尝试调整速度（FASTER/SLOWER/IDLE）以寻找安全间隙。在匝道末端，如果无法变道，宁可减速等待。
            * **主道车：** 在合流区，如果右侧有匝道车需要合流，并且左侧车道安全，考虑左变道避让。如果无法左变道，尝试调整速度（SLOWER/FASTER/IDLE）为匝道车创造空间。
            * **所有车辆：** 保持高效行驶，避免不必要的减速。
        5.  **输出指令：** 以JSON格式返回你的最终决策。你的响应必须始终是JSON，并且包含 `decision` (string), `target_lane` (int), `reasoning` (string) 字段。
            * `decision` 字段必须是 `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER` 中的一个。
            * `target_lane` 应该是目标车道的数字索引 (0, 1, 2, 3)。
            * `reasoning` 字段要详细说明你做出该决策的原因，特别是结合了哪些工具的输出。
        
        **注意：**
        - 如果没有任何安全动作可选，你必须选择最能避免碰撞的 `SLOWER` 动作，并说明原因。
        - 你的输出必须是一个有效的JSON对象。
        """
        return prompt

    def get_decision(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interacts with the LLM to get a driving decision.
        Manages tool calls and parsing of LLM's responses.
        """
        # Scenario update is now called for each individual vehicle within the main loop
        # self.scenario._update_vehicles() # Removed from here, called per vehicle in simulation loop

        messages = [{"role": "system", "content": self.system_prompt}]
        
        user_prompt = f"""
        当前交通场景感知信息如下：
        自车信息: {json.dumps(observation['ego_vehicle'], indent=2)}
        周围车辆信息: {json.dumps(observation['nearby_vehicles'], indent=2)}
        道路信息: {json.dumps(observation['road_info'], indent=2)}
        当前交通密度: {observation.get('traffic_density', 'unknown')}
        
        请分析当前交通态势，并使用你的工具来推断和生成一个最优的合流决策。
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
                        for tool_obj in self.tools.values() # Provide all tools to LLM
                    ],
                    tool_choice="auto", # Let LLM decide whether to call a tool
                    temperature=self.temperature,
                    # Force JSON response only on the last attempt if previous ones weren't JSON
                    response_format={"type": "json_object"} if i == max_tool_calls -1 else None
                )
                
                response_message = response.choices[0].message
                
                if response_message.tool_calls:
                    # LLM wants to call one or more tools
                    messages.append(response_message) # Add LLM's tool call request to messages
                    
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = tool_call.function.arguments # This is a JSON string
                        
                        tool_to_call = self.tools_map.get(function_name)
                        if tool_to_call:
                            try:
                                # Parse tool arguments: handle simple string or JSON string with 'input' key
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
                    # LLM provides a final answer (hopefully JSON)
                    llm_output = response_message.content
                    try:
                        decision = json.loads(llm_output)
                        # Validate the decision format against expected fields (now without target_speed)
                        if all(k in decision for k in ["decision", "target_lane", "reasoning"]):
                            return decision
                        else:
                            print(f"LLM output missing required fields: {llm_output}")
                            raise ValueError("Invalid LLM output format")
                    except json.JSONDecodeError:
                        print(f"LLM did not return valid JSON: {llm_output}. Retrying with force JSON.")
                        # If not JSON, append LLM's text response and continue loop (this will trigger force JSON next time)
                        messages.append({"role": "assistant", "content": llm_output})
            
            except Exception as e:
                print(f"Error during LLM interaction: {e}")
                break # Exit loop on unhandled exception

        # Fallback to a default safe behavior if LLM fails or doesn't return valid JSON after attempts
        print("LLM failed to provide a valid decision after multiple attempts or errors. Falling back to emergency deceleration.")
        # Try to get ego's current speed and lane from the last known observation
        ego_lane_id_int = observation['ego_vehicle']['lane_id'] if 'ego_vehicle' in observation and observation['ego_vehicle'] else -1 

        return {
            "decision": "SLOWER", # Safest fallback action
            "target_lane": ego_lane_id_int,
            "reasoning": "LLM interaction failed, falling back to emergency deceleration."
        }