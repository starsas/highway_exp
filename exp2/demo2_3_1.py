import numpy as np
import copy
import csv
import gym
import highway_env
from moviepy.editor import ImageSequenceClip
from highway_env.envs.common.mdp_controller import mdp_controller
from highway_env.utils import rotated_rectangles_intersect

def predict_vehicle_trajectory(vehicle, action, env_copy, steps=5):
    if vehicle==None:
        return None
    v = copy.deepcopy(vehicle)
    traj = []
    for _ in range(steps):
        mdp_controller(v, env_copy, action)
        traj.append((v.position.copy(), v.heading, v.speed))
    return traj

def check_collision(vehicle, other, other_state):
    if vehicle.crashed or other is vehicle:
        return False
    pos_o = other_state[0]
    if np.linalg.norm(pos_o - vehicle.position) > vehicle.LENGTH:
        return False
    return rotated_rectangles_intersect(
        (vehicle.position, 0.9*vehicle.LENGTH, 0.9*vehicle.WIDTH, vehicle.heading),
        (pos_o, 0.9*other.LENGTH, 0.9*other.WIDTH, other_state[1])
    )

def check_safety(vehicle, action, env_copy):
    traj = predict_vehicle_trajectory(vehicle, action, env_copy)
    # 周围车辆
    v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle)
    side = env_copy.road.network.side_lanes(vehicle.lane_index)
    v_fr, v_rr = (None, None)
    if side:
        v_fr, v_rr = env_copy.road.surrounding_vehicles(vehicle, side[0])
    # 每一步都检查前后碰撞和最小距离
    for t, state in enumerate(traj):
        pos, _, _ = state
        for other, other_traj in ((v_fl, predict_vehicle_trajectory(v_fl, getattr(v_fl, 'action',1), env_copy)),
                                  (v_rl, predict_vehicle_trajectory(v_rl, getattr(v_rl, 'action',1), env_copy)),
                                  (v_fr, predict_vehicle_trajectory(v_fr, getattr(v_fr, 'action',1), env_copy)),
                                  (v_rr, predict_vehicle_trajectory(v_rr, getattr(v_rr, 'action',1), env_copy))):
            if other is None: continue
            ostate = other_traj[t] if t < len(other_traj) else other_traj[-1]
            if check_collision(vehicle, other, ostate):
                return False
            # 前后安全距离
            if abs(ostate[0][0] - pos[0]) < (10 if other in (v_fl, v_rl) else 15):
                return False
    return True

def get_safe_actions(vehicle, env):
    """返回当前车辆所有安全可行动作列表"""
    env_copy = copy.deepcopy(env)
    acts = []
    if vehicle.lane_index[2] > 0: acts.append(0)
    if vehicle.lane_index[2] < 2: acts.append(2)
    if vehicle.speed < vehicle.SPEED_MAX: acts.append(3)
    if vehicle.speed > 0: acts.append(4)
    acts.append(1)
    return [a for a in acts if check_safety(vehicle, a, env_copy)]

def get_dynamic_alpha(vehicle):
    # 匝道 lane_index[2]==2 用 0.3，主路其他车道用 0.7
    return 0.3 if vehicle.lane_index[2] == 2 else 0.7

def stahlberg_two_agent(actions,v1, v2, env):
    """
    两车之间 Ståhlberg 博弈：返回最佳动作对(a1, a2)
    """
    env_copy1 = copy.deepcopy(env)
    ac=copy.deepcopy(actions)
    # 保留收益
    env_copy1.step(actions)
    d1 = env_copy1.controlled_vehicles[v1.id].local_reward
    d2 = env_copy1.controlled_vehicles[v2.id].local_reward
    # 各自安全动作
    A1 = get_safe_actions(v1, env)
    A2 = get_safe_actions(v2, env)
    # A1 = [0,1,2,3,4]
    # A2 = [0,1,2,3,4]
    best_prod = -np.inf
    best_pair = (1,1)
    α1 = get_dynamic_alpha(v1)
    α2 = get_dynamic_alpha(v2)

    # 枚举组合
    for a1 in A1:
        for a2 in A2:
            env_copy2 = copy.deepcopy(env)
            ac[v1.id],ac[v2.id]=a1,a2
            env_copy2.step(ac)
            u1 = env_copy2.controlled_vehicles[v1.id].local_reward
            u2 = env_copy2.controlled_vehicles[v2.id].local_reward
            gain1, gain2 = u1 - d1, u2 - d2
            if gain1 <= 0 or gain2 <= 0: 
                continue
            prod = (gain1**α1) * (gain2**α2)
            if prod > best_prod:
                best_prod = prod
                best_pair = (a1, a2)
    return best_pair

def simulate_and_save_videos(env, num_steps=400,
                             env_video_file="./stahlberg.mp4",
                             output_csv_file="./stahlberg.csv",
                             fps=10):
    action_mapping = {0:"LANE_LEFT",1:"IDLE",2:"LANE_RIGHT",3:"FASTER",4:"SLOWER"}
    env_frames = []
    with open(output_csv_file, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['step','vehicle_id','action','lane','pos','speed','reward'])
        writer.writeheader()
        env.reset()
        for step in range(num_steps):
            # 找到一对合流车辆：一个匝道(2)，一个非匝道
            ramp = [v for v in env.controlled_vehicles if v.lane_index[2]==2]
            main = [v for v in env.controlled_vehicles if v.lane_index[2]!=2]
            actions = [1]*len(env.controlled_vehicles)
            changes = [0]*len(env.controlled_vehicles)
            if ramp and main:
                # 选距离最近的一对
                pairs = [(v1,v2) for v1 in ramp for v2 in main]
                v1,v2 = min(pairs, key=lambda x: np.linalg.norm(x[0].position - x[1].position))
                a1,a2 = stahlberg_two_agent(actions,v1, v2, env)
                actions[v1.id], actions[v2.id] = a1, a2
                changes[v1.id], changes[v2.id] = a1, a2
            # 其余车辆维持 FIFO 逻辑
            for v in env.controlled_vehicles:
                if changes[v.id] == 0:
                    if actions[v.id] == 1:
                        sa = get_safe_actions(v, env)
                        if not sa:  # 检查安全动作列表是否为空
                            actions[v.id] = 1  # 默认保持不动
                        else:
                            if v.lane_index[2] == 2:
                                actions[v.id] = sa[0] if sa[0] in (0, 4) else 4
                            else:
                                actions[v.id] = sa[0] if sa[0] in (3, 1) else 1
            # 渲染 & 迈步
            env_frames.append(env.render(mode='rgb_array'))
            obs, reward, done, info = env.step(actions)
            print(info['agents_rewards'])
            # 记录 CSV
            for v in env.controlled_vehicles:
                r = info["agents_rewards"][v.id] if "agents_rewards" in info else env._agent_reward(actions[v.id], v)
                writer.writerow({
                    'step': step,
                    'vehicle_id': v.id,
                    'action': action_mapping[actions[v.id]],
                    'lane': v.lane_index[2],
                    'pos': v.position[0],
                    'speed': v.speed,
                    'reward': r
                })
            if done: break

    clip = ImageSequenceClip(env_frames, fps=fps)
    clip.write_videofile(env_video_file, codec="libx264")
    print(f"Saved video to {env_video_file}")
    print(f"Logged data to {output_csv_file}")

if __name__ == "__main__":
    env = gym.make('merge-multi-agent-v0')
    simulate_and_save_videos(env)
    env.close()
