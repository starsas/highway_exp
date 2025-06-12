import openai
import json
import numpy as np
import re
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
        self.id = getattr(vehicle, 'id', f"vehicle_{id(vehicle)}")
        self.speed = vehicle.speed
        self.lane_id = vehicle.lane_index[2] if vehicle.lane_index else -1
        self.lanePosition = vehicle.position[0]
        self.original_lane_index = vehicle.lane_index
        self.is_controlled = hasattr(vehicle, 'is_controlled') and vehicle.is_controlled

class Scenario:
    """
    Adapts highway_env.Env to the format expected by the LLM tools.
    This class wraps the environment state, providing a simplified view for LLM tool access.
    """
    def __init__(self, env: Any):
        self.env = env
        self.vehicles: Dict[str, MockVehicle] = {}
        self.lanes: Dict[int, Any] = {}
        self._update_vehicles()
        self._update_lanes()

    def _update_vehicles(self):
        self.vehicles = {}
        if self.env.controlled_vehicles:
            for i, v in enumerate(self.env.controlled_vehicles):
                veh_id = "ego" if i == 0 else getattr(v, 'id', f"vehicle_{id(v)}")
                mock_v = MockVehicle(v)
                mock_v.is_controlled = True
                self.vehicles[veh_id] = mock_v
                v.id = veh_id
        
        for v in self.env.road.vehicles:
            veh_id = getattr(v, 'id', f"vehicle_{id(v)}")
            if veh_id not in self.vehicles:
                mock_v = MockVehicle(v)
                mock_v.is_controlled = False
                self.vehicles[veh_id] = mock_v

    def _update_lanes(self):
        self.lanes = {
            0: type('obj', (object,), {'laneIdx': 0})(),
            1: type('obj', (object,), {'laneIdx': 1})(),
            2: type('obj', (object,), {'laneIdx': 2})(),
            3: type('obj', (object,), {'laneIdx': 3})()
        }

def is_in_merging_area(vehicle):
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

# --- Tool Classes (These remain as they were in your previous correct version) ---

@prompts(name='Get_Available_Actions',
             description="""Useful to know what available actions (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER) the ego vehicle can take in this situation. Call this tool with input 'ego'.""")
class getAvailableActions:
    def __init__(self, env: Any) -> None:
        self.env = env
    def inference(self, input: str) -> str:
        availableActions = list(ACTIONS_ALL.keys())
        outputPrefix = 'You can ONLY use one of the following actions: \n'
        for action_idx in availableActions:
            outputPrefix += f"{ACTIONS_ALL[action_idx]}--{ACTIONS_DESCRIPTION[action_idx]}; \n"
        outputPrefix += """\nTo check decision safety you should follow steps:
        Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle action affect the current lane, while left and right lane changes affect the corresponding lane.
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
        Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle affect the current lane, while left and right lane changes affect the corresponding lane.
        Step 2:(Optional) Get the vehicles in this lane that you may affect, ONLY when you don't know.
        Step 3: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
        Follow the instructions and remember to use the proper tools mentioned in the tool list once a time.
        """

@prompts(name='Get_Available_Lanes',
             description="""Useful to know the available lanes (current, left, right) for a vehicle given the road network and merging rules. Input: vehicle ID (e.g., 'ego').""")
class getAvailableLanes:
    def __init__(self, env: Any) -> None:
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
        if current_lane_idx > 0:
            if current_lane_idx == 3:
                if is_in_merging_area(MockVehicle(ego_vehicle_obj)):
                    left_lane_id = 2
                    available_lanes_info.append(f"`lane_{left_lane_id}` is to the left of the current lane (merge lane), permitted due to being in merging area.")
                else:
                    available_lanes_info.append(f"Cannot change to `lane_2` from current lane `lane_3` as vehicle is not in merging area.")
            else:
                left_lane_id = current_lane_idx - 1
                available_lanes_info.append(f"`lane_{left_lane_id}` is to the left of the current lane.")
        if current_lane_idx < 3:
            if current_lane_idx == 2:
                if is_in_merging_area(MockVehicle(ego_vehicle_obj)):
                    right_lane_id = 3
                    available_lanes_info.append(f"`lane_{right_lane_id}` is to the right of the current lane (merge lane), permitted due to being in merging area.")
                else:
                    available_lanes_info.append(f"Cannot change to `lane_3` from current lane `lane_2` as vehicle is not in merging area.")
            elif current_lane_idx < 2:
                right_lane_id = current_lane_idx + 1
                available_lanes_info.append(f"`lane_{right_lane_id}` is to the right of the current lane.")
        return f"The available lanes of `{vid}` are: " + " ".join(available_lanes_info)

@prompts(name='Get_Lane_Involved_Car',
             description="""Useful to identify leading and trailing vehicles in a specific lane relative to the ego vehicle. Input: lane ID (e.g., 'lane_0', 'lane_1', 'lane_2', 'lane_3').""")
class getLaneInvolvedCar:
    def __init__(self, env: Any) -> None:
        self.env = env
    def inference(self, laneID_str: str) -> str:
        try:
            lane_idx = int(laneID_str.split('_')[1])
        except (IndexError, ValueError):
            return "Not a valid lane id format (expected 'lane_X')! Make sure you have use tool `Get_Available_Lanes` first."
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
        output_vehicles_info = []
        for v in self.env.road.vehicles:
            if v.id != ego_vehicle_obj.id and v.lane_index[2] == lane_idx:
                distance = v.position[0] - ego_vehicle_obj.position[0]
                vehicle_type = "CAV" if hasattr(v, 'is_controlled') and v.is_controlled else "HDV"
                vehicle_info = f"Vehicle {v.id} (Type: {vehicle_type}, Speed: {round(v.speed, 1)}m/s)"
                if distance > 0:
                    if distance < min_front_distance:
                        min_front_distance = distance
                        leadingCar = v
                        output_vehicles_info.append(f"{vehicle_info} is driving in front of ego car for {round(distance, 2)} meters.")
                else:
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

@prompts(name='Is_Change_Lane_Conflict_With_Car',
             description="""Useful to check if a lane change action to a specific lane conflicts with a specific car. Input: comma-separated string 'lane_id,vehicle_id' (e.g., 'lane_2,vehicle_5').""")
class isChangeLaneConflictWithCar:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
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
        if (ego_vehicle_obj.lane_index[2] == 3 and lane_idx == 2) or \
           (ego_vehicle_obj.lane_index[2] == 2 and lane_idx == 3):
            if not is_in_merging_area(MockVehicle(ego_vehicle_obj)):
                return f"Change lane from `lane_{ego_vehicle_obj.lane_index[2]}` to `lane_{lane_idx}` is not permitted as vehicle is not in the merging area."
        if target_vehicle_obj.lane_index[2] != lane_idx:
            return f"Vehicle `{vid}` is not in `lane_{lane_idx}`. Please check the lane ID again."
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]
        if distance > 0:
            relativeSpeed = ego_vehicle_obj.speed - target_vehicle_obj.speed
            if distance - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Change lane to `lane_{lane_idx}` is safe with `{vid}` (leading)."
            else:
                return f"Change lane to `lane_{lane_idx}` may be conflict with `{vid}` (leading), which is unacceptable. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        else:
            relativeSpeed = target_vehicle_obj.speed - ego_vehicle_obj.speed
            if abs(distance) - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Change lane to `lane_{lane_idx}` is safe with `{vid}` (rearing)."
            else:
                return f"Change lane to `lane_{lane_idx}` may be conflict with `{vid}` (rearing), which is unacceptable. Distance: {abs(distance):.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."

@prompts(name='Is_Acceleration_Conflict_With_Car',
             description="""Useful to check if accelerating conflicts with a specific car in the same lane. Input: vehicle ID (e.g., 'vehicle_5').""")
class isAccelerationConflictWithCar:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0
        self.acceleration = 4.0
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
            return f'{vid} is not in the same lane with ego, please call `Get_Lane_Involved_Car` and rethink your input.'
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]
        if distance > 0:
            relativeSpeed = (ego_vehicle_obj.speed + self.acceleration) - target_vehicle_obj.speed
            if distance - (self.VEHICLE_LENGTH * 2) > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Acceleration is safe with `{vid}`."
            else:
                return f"Acceleration may be conflict with `{vid}`, which is unacceptable. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        else:
            return f"Acceleration is safe with {vid} (rearing)."

@prompts(name='Is_Keep_Speed_Conflict_With_Car',
             description="""Useful to check if maintaining current speed conflicts with a specific car in the same lane. Input: vehicle ID (e.g., 'vehicle_5').""")
class isKeepSpeedConflictWithCar:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0
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
            return f'{vid} is not in the same lane with ego, please call `Get_Lane_Involved_Car` and rethink your input.'
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]
        if distance > 0:
            relativeSpeed = ego_vehicle_obj.speed - target_vehicle_obj.speed
            if distance - (self.VEHICLE_LENGTH * 2) > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Keep lane with current speed is safe with {vid}"
            else:
                return f"Keep lane with current speed may be conflict with {vid}, you need to consider decelerate. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        else:
            return f"Keep lane with current speed is safe with {vid} (rearing)."

@prompts(name='Is_Deceleration_Safe',
             description="""Useful to check if decelerating conflicts with a specific car in the same lane. Input: vehicle ID (e.g., 'vehicle_5').""")
class isDecelerationSafe:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
        self.deceleration = 3.0
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
            return f'{vid} is not in the same lane with ego, please call `Get_Lane_Involved_Car` and rethink your input.'
        distance = target_vehicle_obj.position[0] - ego_vehicle_obj.position[0]
        if distance > 0:
            relativeSpeed = (ego_vehicle_obj.speed - self.deceleration) - target_vehicle_obj.speed
            if distance - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"Deceleration with current speed is safe with {vid}"
            else:
                return f"Deceleration with current speed may be conflict with {vid}, if you have no other choice, slow down as much as possible. Distance: {distance:.2f}m, Relative Speed: {relativeSpeed:.2f}m/s."
        else:
            return f"Deceleration with current speed is safe with {vid} (rearing)."

# --- 2. LLM Agent Class ---

class LLMAgent:
    def __init__(self, api_key: str, env: Any, model_name: str = "deepseek-coder", temperature: float = 0.7):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name
        self.temperature = temperature
        self.env = env
        self.scenario = Scenario(self.env) # This will store ALL vehicles, as environment is global

        self.tools = {
            getAvailableActions(self.env),
            isActionSafe(),
            getAvailableLanes(self.env),
            getLaneInvolvedCar(self.env),
            isChangeLaneConflictWithCar(self.env),
            isAccelerationConflictWithCar(self.env),
            isKeepSpeedConflictWithCar(self.env),
            isDecelerationSafe(self.env)
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
        You are an autonomous driving agent making decisions in a highway merging area. Your task is to provide the optimal merging strategy for a Connected and Automated Vehicle (CAV, self-vehicle, ID is 'ego').

        **Scenario Description:**
        We are in a highway entrance area with **three main lanes** (numbered 0, 1, 2, from left to right) and **one ramp lane** (numbered 3, merging from the right side of main lane 2).
        The traffic flow is **mixed-type**, consisting of Human-Driven Vehicles (HDVs) and Connected and Automated Vehicles (CAVs).
        - **CAV Information Sharing:** You, as a CAV, can query for precise state information (position, speed, lane, etc.) for all vehicles on the road by using the provided tools. You DO NOT have an initial full list of nearby vehicles; you must actively discover them using tools.
        - **HDV Observation Only:** You can only indirectly query for HDV information via tools. You cannot directly communicate with them or obtain their future intentions. You need to predict their possible actions based on their current behavior.

        **Your Core Decision Objective:**
        While ensuring **absolute safety** as the top priority, **maximize traffic efficiency in the merging area and the ego vehicle's own throughput efficiency**. This entails:
        1.  Maintaining reasonable speeds during merging, avoiding unnecessary stops or excessively slow driving.
        2.  Prioritizing smooth integration into main lanes to minimize impact on existing traffic flow.
        3.  Minimizing waiting time on the ramp.
        4.  Ensuring driving comfort by avoiding abrupt accelerations or decelerations.

        **Traffic Rules & General Driving Principles:**
        * Try to keep a safe distance to the car in front of you.
        * If there is no safe decision for any action, always resort to slowing down as an emergency measure.
        * DONOT change lane frequently. If you need to change lane, double-check the safety of vehicles on target lane.
        

         **Special Merging Area Rules:**
        * Lane changes between main lanes and the ramp (lane 2 and lane 3) are ONLY permitted when the vehicle is physically located within the merging area (i.e., main lane 2 or ramp lane 3 near the merge point). Lane changes between other non-merging lanes are not subject to this specific restriction.
        * 

        **Your Action Space:**
        - LANE_LEFT (0): Change lane to the left.
        - IDLE (1): Remain in the current lane with current speed.
        - LANE_RIGHT (2): Change lane to the right.
        - FASTER (3): Accelerate the vehicle.
        - SLOWER (4): Decelerate the vehicle.
        
        **Available Tools:**
        The following are the tools you can use. You must strictly follow their names and descriptions.
        {tool_descriptions}
        
        **Decision-Making Process Guide:**
        You must act as a sophisticated planner. Your process must involve:
        1.  **Initial Perception:** Start by understanding your `ego_vehicle` information and `road_info`.
        2.  **Tool-Driven Perception:** Systematically use tools to gather more information. You should prioritize:
            * Using `Get_Available_Actions` to understand your options.
            * Using `Get_Available_Lanes` to understand potential target lanes.
            * For each relevant lane (current, left, right), use `Get_Lane_Involved_Car` to identify other vehicles in those lanes.
            * For each identified vehicle and potential action, use the specific safety check tools (e.g., `Is_Change_Lane_Conflict_With_Car`, `Is_Acceleration_Conflict_With_Car`, `Is_Keep_Speed_Conflict_With_Car`, `Is_Deceleration_Safe`) to assess safety.
            * **HDV Behavior Prediction:** For HDVs, assume they will follow basic traffic rules and driving habits. In the absence of clear intentions, tend to drive conservatively, e.g., maintaining current lane and speed, unless there's an obstacle ahead.
        3.  **Safety First:** After gathering information, thoroughly assess the safety of all possible actions. Safety is paramount; an unsafe action must never be chosen.
        4.  **Optimal Strategy Selection:** From the safe actions, select the one that best meets the efficiency and comfort objectives.
            * **Ramp Vehicle (CAV):** In the merging area, if a safe lane change to the main lane is possible, prioritize it. If not, adjust speed (FASTER/SLOWER/IDLE) to find/create a gap, aiming for faster merge. If at ramp end and no merge, decelerate and wait.
            * **Main Lane Vehicle (CAV):** In merging area, if ramp vehicle needs to merge and left lane is safe, consider changing left to aid flow. If not, adjust speed to create a gap, avoiding large speed drops.
            * **All CAVs:** In non-merging areas, drive efficiently.
        5.  **Output Instruction:** Return your final decision in JSON format. Your response must always be JSON and include the `decision` (string) and `reasoning` (string) fields.
            * The `decision` field must be one of `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`.
            * The `reasoning` field should explain in detail why you made this decision, specifically referencing the outputs of the tools used.

        **Your response MUST always be a JSON object, particularly for your FINAL decision.**

        **Note:**
        - If no safe action is available after all checks, you MUST choose the `SLOWER` action as an emergency evasive maneuver and explain why.
        """
        return prompt

    def get_decision(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Initial user prompt with only ego vehicle and road info
        user_prompt_initial = f"""
        Current ego vehicle information: {json.dumps(observation['ego_vehicle'], indent=2)}
        Current road information: {json.dumps(observation['road_info'], indent=2)}
        Current traffic density: {observation.get('traffic_density', 'unknown')}
        
        Please begin your decision-making process by calling the appropriate tools to gather all necessary environmental perception and safety information.
        """
        messages.append({"role": "user", "content": user_prompt_initial})

        max_iterations = 15 # Set a higher limit for tool calls to allow complex reasoning

        for i in range(max_iterations):
            try:
                # LLM decides whether to call a tool or provide a final answer
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=[
                        {"type": "function", "function": {"name": tool_obj.name, "description": tool_obj.description, "parameters": {"type": "object", "properties": {"input": {"type": "string"}}}}}
                        for tool_obj in self.tools
                    ],
                    tool_choice="auto", # LLM decides
                    temperature=self.temperature,
                    response_format={"type": "json_object"} if i == max_iterations -1 else {"type": "text"} # Force JSON only on last iteration if needed
                )
                
                response_message = response.choices[0].message
                # print(f"LLM Response (Iteration {i+1}): {response_message}")
                
                if response_message.tool_calls:
                    messages.append(response_message) # Add LLM's tool call request
                    
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args_str = tool_call.function.arguments # This is a JSON string
                        
                        tool_to_call = self.tools_map.get(function_name)
                        if tool_to_call:
                            try:
                                # Parse tool arguments: handle JSON string to dict, then get 'input'
                                parsed_args = json.loads(function_args_str)
                                tool_input = parsed_args.get('input') # Extract 'input'
                                
                                # Execute the tool
                                tool_output_content = tool_to_call.inference(tool_input)
                                
                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_output_content,
                                })
                                # print(f"Tool Call Executed: {function_name}(input='{tool_input}') -> Output: {tool_output_content}")

                            except json.JSONDecodeError:
                                error_msg = f"Invalid JSON arguments for tool {function_name}: {function_args_str}"
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(error_msg)
                            except KeyError:
                                error_msg = f"Missing 'input' key in arguments for tool {function_name}: {function_args_str}"
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(error_msg)
                            except Exception as e:
                                error_msg = f"Error executing tool {function_name} with args {function_args_str}: {e}"
                                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                                print(error_msg)
                        else:
                            error_msg = f"Error: Tool '{function_name}' not found in tools_map."
                            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_msg})
                            print(error_msg)
                else:
                    # LLM provided a final answer (hopefully JSON) or a text response
                    llm_output_raw = response_message.content
                    
                    # Try to extract JSON from markdown fences, or use as-is
                    json_match = re.search(r'```json\n(.*?)```', llm_output_raw, re.DOTALL)
                    if json_match:
                        clean_llm_output = json_match.group(1).strip()
                    else:
                        clean_llm_output = llm_output_raw.strip()
                    
                    try:
                        decision = json.loads(clean_llm_output)
                        # Check required fields: decision and reasoning
                        if all(k in decision for k in ["decision", "reasoning"]):
                            return decision # Found valid JSON decision, return it
                        else:
                            print(f"LLM output is JSON but missing required fields: {llm_output_raw}")
                            # If not a complete decision, append LLM's response and continue
                            messages.append({"role": "assistant", "content": llm_output_raw})
                            continue # Continue loop for more iterations
                    except json.JSONDecodeError:
                        print(f"LLM did not return valid JSON. Appending as text and continuing: {llm_output_raw}")
                        # If not JSON, append LLM's response as text and continue loop
                        messages.append({"role": "assistant", "content": llm_output_raw})
                        continue # Continue loop for more iterations

            except Exception as e:
                print(f"Error during LLM interaction (outer loop): {e}")
                return self._fallback_decision(observation, reason=f"An unexpected error occurred during LLM process: {e}")

        # If max_iterations reached without a valid decision
        print(f"Max iterations ({max_iterations}) reached without LLM providing a valid JSON decision.")
        return self._fallback_decision(observation, reason=f"Max tool calls ({max_iterations}) reached, no valid JSON decision.")

    def _fallback_decision(self, observation: Dict[str, Any], reason: str = "LLM interaction failed, falling back to emergency deceleration.") -> Dict[str, Any]:
        print(reason)
        # Fallback decision now matches the required output format (no target_lane)
        return {
            "decision": "SLOWER",
            "reasoning": reason
        }