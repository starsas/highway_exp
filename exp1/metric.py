import pandas as pd
import numpy as np
import json # For parsing potential JSON in reasoning if needed
import re   # For parsing string IDs or messages

# --- Constants for Lane/Road Structure (Should match your environment setup) ---
# These constants should be consistent with custom_llm.py and your environment's _make_road
NUM_MAIN_LANES = 2 
MAIN_LANE_INDICES = [0, 1]
RAMP_LANE_IDX = 2 
MERGE_MAIN_LANE_IDX = 1 
MERGE_RAMP_LANE_IDX = 2 

# --- Constants for Analysis Thresholds ---
EXIT_X_THRESHOLD = 515  # X-coordinate to consider vehicle as having exited the road
HARD_ACCEL_THRESHOLD = 3.0 # m/s^2 for "hard" acceleration/deceleration
MIN_THW_THRESHOLD = 1.0 # seconds, for identifying unsafe THW


def analyze_simulation_data(csv_file_path: str, num_steps: int, fps: int):
    """
    Analyzes the simulation data from a CSV file to calculate various metrics.

    Args:
        csv_file_path (str): The path to the CSV file containing simulation data.
        num_steps (int): The total number of simulation steps performed.
        fps (int): The frames per second used for the simulation, used for time calculations.
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded data from {csv_file_path}. Shape: {df.shape}")
        print("Columns available:", df.columns.tolist())
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Ensure necessary columns exist
    required_cols = ['step', 'vehicle_id', 'lane_index', 'position', 'speed', 'llm_decision', 'reasoning']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV is missing one or more required columns: {required_cols}")
        return

    # Convert lane_index string representation to integer if stored as string "(c, d, 0)" -> 0
    # Assuming lane_index in CSV is stored as integer id directly (e.g., 0, 1, 2)
    # If it's a string like "('c', 'd', 0)", you'll need to parse it:
    # df['lane_index_id'] = df['lane_index'].apply(lambda x: int(x.strip().split(',')[-1].replace(')', '').strip()))
    # For now, assuming 'lane_index' column already contains the numerical ID directly
    df['lane_index_id'] = df['lane_index'] # Use directly if already numerical

    # --- 1. 平均车速 (Average Speed) ---
    print("\n--- 1. 平均车速分析 ---")
    average_speed_overall = df['speed'].mean()
    print(f"所有车辆在所有步骤中的平均速度: {average_speed_overall:.2f} m/s")
    
    average_speed_per_vehicle = df.groupby('vehicle_id')['speed'].mean()
    print("\n各车辆的平均速度:")
    print(average_speed_per_vehicle.round(2))

    # --- 2. 碰撞次数 (Number of Collisions) ---
    print("\n--- 2. 碰撞分析 (基于LLM推理原因的间接判断) ---")
    collision_keywords = 'collision|unsafe|crash|fallback|emergency deceleration|error'
    collision_related_decisions = df[df['reasoning'].str.contains(collision_keywords, case=False, na=False)]
    
    if not collision_related_decisions.empty:
        print(f"检测到 {len(collision_related_decisions)} 条与碰撞/紧急情况相关的决策记录。")
        print("部分相关记录:")
        print(collision_related_decisions[['step', 'vehicle_id', 'llm_decision', 'reasoning']].head())
        print("\n注意: 这是一个间接判断。要精确统计碰撞，需要仿真环境提供直接的碰撞标志。")
    else:
        print("未在 'reasoning' 列中检测到明显的碰撞或紧急减速指示。")

    # --- 3. 动作分布 (Action Distribution) ---
    print("\n--- 3. 动作分布分析 ---")
    action_distribution = df['llm_decision'].value_counts(normalize=True) * 100
    print("LLM决策动作分布 (%):")
    print(action_distribution.round(2))

    print("\n各车辆的决策动作分布 (%):")
    action_distribution_per_vehicle = df.groupby('vehicle_id')['llm_decision'].value_counts(normalize=True) * 100
    print(action_distribution_per_vehicle.unstack(fill_value=0).round(2))

    # --- 4. 吞吐量分析 (Throughput Analysis) ---
    print("\n--- 4. 吞吐量分析 ---")
    total_simulation_time_seconds = num_steps / fps
    if total_simulation_time_seconds > 0:
        max_position_per_vehicle = df.groupby('vehicle_id')['position'].max()
        vehicles_passed_exit = max_position_per_vehicle[max_position_per_vehicle >= EXIT_X_THRESHOLD]
        num_vehicles_passed = len(vehicles_passed_exit)
        
        throughput_vehicles_per_second = num_vehicles_passed / total_simulation_time_seconds
        throughput_vehicles_per_hour = throughput_vehicles_per_second * 3600
        
        print(f"通过出口 (X >= {EXIT_X_THRESHOLD:.0f}m) 的车辆数量: {num_vehicles_passed}")
        print(f"总仿真时间: {total_simulation_time_seconds:.2f} 秒")
        print(f"吞吐量 (辆/秒): {throughput_vehicles_per_second:.2f}")
        print(f"吞吐量 (辆/小时): {throughput_vehicles_per_hour:.2f}")
    else:
        print("仿真时间为零或负，无法计算吞吐量。")

    # --- 5. 平均加速度/减速度 (Average Acceleration/Deceleration) ---
    print("\n--- 5. 平均加速度/减速度分析 ---")
    df['speed_prev'] = df.groupby('vehicle_id')['speed'].shift(1)
    df['delta_speed'] = df['speed'] - df['speed_prev']
    dt_sim = 1 / fps # Time step per frame/step

    # Calculate acceleration (m/s^2)
    df['acceleration'] = df['delta_speed'] / dt_sim

    # Filter out the first step for each vehicle (where speed_prev is NaN)
    df_accel_valid = df.dropna(subset=['acceleration'])

    avg_acceleration_positive = df_accel_valid[df_accel_valid['acceleration'] > 0]['acceleration'].mean()
    avg_deceleration_negative = df_accel_valid[df_accel_valid['acceleration'] < 0]['acceleration'].mean() # Keep as negative for 'deceleration' sense

    print(f"平均正加速度 (加速): {avg_acceleration_positive:.2f} m/s^2")
    print(f"平均负加速度 (减速): {avg_deceleration_negative:.2f} m/s^2")

    # Count hard acceleration/deceleration events
    hard_accel_events = df_accel_valid[df_accel_valid['acceleration'] >= HARD_ACCEL_THRESHOLD].shape[0]
    hard_decel_events = df_accel_valid[df_accel_valid['acceleration'] <= -HARD_ACCEL_THRESHOLD].shape[0]
    
    print(f"急加速事件 ({HARD_ACCEL_THRESHOLD:.1f} m/s^2 以上): {hard_accel_events} 次")
    print(f"急减速事件 ({-HARD_ACCEL_THRESHOLD:.1f} m/s^2 以下): {hard_decel_events} 次")

    # --- 6. 车头时距 (Time Headway - THW) ---
    print("\n--- 6. 车头时距 (THW) 分析 ---")
    df_thw = df.copy() # Use a copy to avoid modifying original df for subsequent calculations

    # Sort data by step, lane_index_id, and position to easily find front vehicle
    df_thw.sort_values(by=['step', 'lane_index_id', 'position'], inplace=True)

    # Find the leading vehicle's position for each vehicle in the same lane at each step
    # Group by step and lane, then shift 'position' to get the leading car's position
    # The shift is applied to the current group (lane_index_id and step)
    df_thw['leading_position'] = df_thw.groupby(['step', 'lane_index_id'])['position'].shift(-1)
    df_thw['leading_speed'] = df_thw.groupby(['step', 'lane_index_id'])['speed'].shift(-1)
    df_thw['leading_id'] = df_thw.groupby(['step', 'lane_index_id'])['vehicle_id'].shift(-1)

    # Calculate longitudinal distance to leading vehicle
    df_thw['distance_to_leading'] = df_thw['leading_position'] - df_thw['position']

    # Calculate Time Headway (THW)
    # Only calculate if vehicle speed is positive and leading vehicle exists and is in front
    # Avoid division by zero
    df_thw['thw'] = df_thw.apply(
        lambda row: row['distance_to_leading'] / row['speed'] if row['speed'] > 0.1 and row['distance_to_leading'] > 0 else np.nan,
        axis=1
    )
    
    # Filter out THW values that are NaN (no leading vehicle or zero speed) or very large (far away)
    valid_thw = df_thw['thw'].dropna()
    valid_thw = valid_thw[valid_thw < 100] # Cap THW for statistical relevance (e.g., ignore if > 100s)

    if not valid_thw.empty:
        avg_thw = valid_thw.mean()
        min_thw = valid_thw.min()
        thw_violations = valid_thw[valid_thw < MIN_THW_THRESHOLD].shape[0]

        print(f"有效车头时距 (THW) 记录数: {valid_thw.shape[0]}")
        print(f"平均车头时距: {avg_thw:.2f} 秒")
        print(f"最小车头时距: {min_thw:.2f} 秒")
        print(f"车头时距小于 {MIN_THW_THRESHOLD:.1f} 秒的违规次数: {thw_violations} 次")
        print("注意: 仅计算了同车道前车的THW。跨车道安全距离需要更复杂的几何计算。")
    else:
        print("没有有效的车头时距数据可供分析 (可能车流量太小或没有前车)。")
    
    # --- 7. 车道保持/变道率 (Lane Keeping / Lane Change Rate) ---
    print("\n--- 7. 车道保持/变道率分析 ---")
    # Identify lane changes for each vehicle
    df['lane_change'] = df.groupby('vehicle_id')['lane_index_id'].diff().fillna(0).astype(bool)
    
    # Count total unique vehicles for normalization
    total_vehicles = df['vehicle_id'].nunique()
    
    # Total recorded steps for each vehicle (approximate duration of presence)
    vehicle_steps = df.groupby('vehicle_id')['step'].count()

    total_lane_changes = df['lane_change'].sum()
    
    if total_vehicles > 0:
        # Average lane changes per vehicle
        avg_lane_changes_per_vehicle = df.groupby('vehicle_id')['lane_change'].sum().mean()
        
        # Average lane change frequency (e.g., changes per 100 steps of presence)
        # Sum of changes / sum of steps for vehicles / (number of vehicles)
        lane_change_frequency_overall = total_lane_changes / (df.shape[0] / total_vehicles) if df.shape[0] > 0 else 0
        
        print(f"总变道次数: {total_lane_changes} 次")
        print(f"每辆车平均变道次数 (在其存在期间): {avg_lane_changes_per_vehicle:.2f} 次")
        # Example of how often a vehicle makes a change, normalized by total steps and vehicles
        print(f"平均变道频率 (总变道数 / 平均车辆存在步数): {lane_change_frequency_overall:.4f} 次/步")
        
        print("\n车辆个体变道次数:")
        print(df.groupby('vehicle_id')['lane_change'].sum())
        
        # Note: Merging success rate is calculated separately.
    else:
        print("没有车辆数据可供车道保持/变道率分析。")

    # --- 8. 合流成功率 (Merging Success Rate) ---
    print("\n--- 8. 合流成功率分析 ---")
    # Identify ramp vehicles (lane is RAMP_LANE_IDX at any point)
    ramp_vehicles_df = df[df['lane_index_id'] == RAMP_LANE_IDX].copy()
    
    unique_ramp_vehicles_ever = ramp_vehicles_df['vehicle_id'].unique()
    num_ramp_vehicles_entered_system = len(unique_ramp_vehicles_ever)
    
    successful_merges = 0
    if num_ramp_vehicles_entered_system > 0:
        for veh_id in unique_ramp_vehicles_ever:
            # Check if this ramp vehicle ever reached the merge main lane AND passed the exit threshold
            # Assumption: Merging happens to MERGE_MAIN_LANE_IDX
            veh_data = df[df['vehicle_id'] == veh_id]
            
            # Check if it ever was in MERGE_MAIN_LANE_IDX
            was_in_main_merge_lane = (veh_data['lane_index_id'] == MERGE_MAIN_LANE_IDX).any()
            
            # Check if it passed the exit threshold while in a main lane
            passed_exit_in_main_lane = (veh_data[veh_data['lane_index_id'].isin(MAIN_LANE_INDICES)]['position'] >= EXIT_X_THRESHOLD).any()
            
            if was_in_main_merge_lane and passed_exit_in_main_lane:
                successful_merges += 1
        
        merging_success_rate = (successful_merges / num_ramp_vehicles_entered_system) * 100
        print(f"进入系统的匝道车辆总数: {num_ramp_vehicles_entered_system}")
        print(f"成功合流并退出系统的匝道车辆数量: {successful_merges}")
        print(f"合流成功率: {merging_success_rate:.2f}%")
    else:
        print("没有匝道车辆进入系统，无法进行合流成功率分析。")

    # --- 9. LLM 决策复杂性/效率 (不在CSV中，提供思路) ---
    print("\n--- 9. LLM 决策复杂性/效率分析 (需要额外日志) ---")
    print("这些指标（如平均工具调用次数、平均对话轮次、错误/回退率）不直接存储在CSV文件中。")
    print("需要修改 custom_llm.py 中的 LLMAgent.get_decision 方法：")
    print("  - 在每次调用LLM和接收其响应时，记录当前迭代次数。")
    print("  - 在工具被调用时，记录工具名称和参数。")
    print("  - 在 LLM 返回最终决策或回退时，记录总迭代次数和是否回退。")
    print("  - 将这些信息存储到另一个日志文件或内存结构中，再进行分析。")

    print("\n--- 所有分析完成 ---")

if __name__ == "__main__":
    # 请确保将 'llm_merge_control.csv' 替换为你实际的CSV文件路径
    # 这里的 num_steps 和 fps 应该与你运行仿真时设置的一致
    output_csv_file_path = "./llm_merge_control.csv" 
    SIM_NUM_STEPS = 400 # 假设你的仿真运行了400步
    SIM_FPS = 10      # 假设你的仿真帧率为10

    analyze_simulation_data(output_csv_file_path, num_steps=SIM_NUM_STEPS, fps=SIM_FPS)