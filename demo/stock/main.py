from tracemalloc import start
import numpy as np
import gym
import random
import torch
import time
import matplotlib.pyplot as plt
from environ import StocksEnv
from data import load_relative


import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__'))))
sys.path.append(root_path)

from models.dueling_dqn import DuelingDQN
from utils.replay_buffer import ReplayBuffer

lr = 1e-2
num_episodes = 100
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

stock_data = {'YNDX': load_relative('..\..\data\stock\YNDX_160101_161231.csv')}
env = StocksEnv(stock_data)
env.seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

replay_buffer = ReplayBuffer(buffer_size)
agent = DuelingDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

start_time = time.time()

return_list = []
smooth_return_list = []
max_q_value_list = []
max_q_value = 0
for i in range(10):
    for i_episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理
            max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                    batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)
        
        return_list.append(episode_return)
        
        if len(smooth_return_list) == 0:
            smooth_return_list.append(episode_return)
        else:
            smooth_return_list.append(smooth_return_list[-1] * 0.9 + episode_return * 0.1)

        progress = i_episode + i * num_episodes + 1
        if progress % 10 == 0:
            print(f'progress: {progress}/{num_episodes * 10}, elapse: {time.time() - start_time}, return: {np.mean(return_list[-10:])}')


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.plot(episodes_list, smooth_return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dueling DQN on StockEnv')
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('Dueling DQN on StockEnv')
plt.show()