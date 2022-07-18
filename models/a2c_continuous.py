import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils.const import *
import gym
import pybullet_envs

HID_SIZE = 128
ENV_ID = 'MinitaurBulletEnv-v0'

# 策略为一个高斯分布，这里计算其对数形式
def calc_logprob(mu_v, var_v, actions_v):
    p1 = -((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))  # torch.clamp()函数用来防止返回的方差太小时将除以0
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2

class ContinuousActionSpaceModel(nn.Module):

    def __init__(self, obs_size, act_size):
        super(ContinuousActionSpaceModel, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus()
        )

        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class AgentA2CContinuous:

    def __init__(self, state_dim, action_dim):
        self.net = ContinuousActionSpaceModel(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def take_action(self, states):
        batch_states = torch.tensor(np.array([states]), dtype=torch.float).to(DEVICE)
        mu_vs, var_vs, _ = self.net(batch_states)
        mus = mu_vs.data.to(torch.device('cpu')).numpy()
        sigmas = torch.sqrt(var_vs).data.to(torch.device('cpu')).numpy()
        actions = np.random.normal(mus, sigmas)
        actions = np.clip(actions, -1, 1)
        return actions[0]

    def update(self, transition_info):
        batch_states = torch.tensor(np.array(transition_info['states']), dtype=torch.float).to(DEVICE)
        batch_actions = torch.tensor(np.array(transition_info['actions']), dtype=torch.float).to(DEVICE)
        batch_rewards = torch.tensor(np.array(transition_info['rewards']), dtype=torch.float).view(-1, 1).to(DEVICE)
        batch_next_states = torch.tensor(np.array(transition_info['next_states']), dtype=torch.float).to(DEVICE)
        batch_dones = torch.tensor(np.array(transition_info['dones']), dtype=torch.float).view(-1, 1).to(DEVICE)
        
        _, _, next_value_vs = self.net(batch_next_states)
        td_target = batch_rewards + GAMMA * next_value_vs
        mu_vs, var_vs, value_vs = self.net(batch_states)
        critic_loss = torch.mean(F.mse_loss(value_vs, td_target.detach())) 

        td_delta = td_target - value_vs
        log_probs = calc_logprob(mu_vs, var_vs, batch_actions)
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


if __name__ == '__main__':    
    env = gym.make(ENV_ID)
    env.reset()
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = AgentA2CContinuous(state_dim, action_dim)
    num_episodes = 100
    start_time = time.time()
    return_list = []
    for i in range(EPOCH_NUM):
        for i_episode in range(num_episodes):
            episode_return = 0
            transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                ho = env.step(action)
                next_state, reward, done, _ = ho
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                
                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            agent.update(transition_dict)
            
            progress = i_episode + i * num_episodes + 1
            if progress % 10 == 0:
                print(f'progress: {progress} / {num_episodes * EPOCH_NUM} | elapse: {time.time() - start_time:.3f}s | return: {np.mean(return_list[-10:])}')


    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(ENV_ID))
    plt.show()