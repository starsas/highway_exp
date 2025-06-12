import gym
import highway_env

env = gym.make('merge-multi-agent-v0')
env.reset()

while True:
    env.render()