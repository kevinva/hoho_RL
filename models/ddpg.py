import random
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym
import pybullet_envs

import os
import sys
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)

from utils.config import *
from utils.replay_buffer import *

ENV_ID = 'MinitaurBulletEnv-v0'
HIDDEN_SIZE = 64

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class TwoLayerFC(torch.nn.Module):
    # 这是一个简单的两层神经网络
    def __init__(self,
                 num_in,
                 num_out,
                 hidden_dim,
                 activation=F.relu,
                 out_fn=lambda x: x):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)

        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, num_in_actor, num_out_actor, num_in_critic, hidden_dim,
                 discrete, action_bound, sigma, actor_lr, critic_lr, tau,
                 gamma, device):
        # self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        # self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        out_fn = (lambda x: x) if discrete else (
            lambda x: torch.tanh(x) * action_bound)
        self.actor = TwoLayerFC(num_in_actor,
                                num_out_actor,
                                hidden_dim,
                                activation=F.relu,
                                out_fn=out_fn).to(device)
        self.target_actor = TwoLayerFC(num_in_actor,
                                       num_out_actor,
                                       hidden_dim,
                                       activation=F.relu,
                                       out_fn=out_fn).to(device)
        self.critic = TwoLayerFC(num_in_critic, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(num_in_critic, 1,
                                        hidden_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = num_out_actor
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.actor(state)[0].clamp(min=-1, max=1)
        # 给动作添加噪声，增加探索
        action = action.detach().numpy() + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)

        # critic网络像更新DQN一样，只是动作由target_actor输出
        # 注意target_critic搭配target_actor使用
        next_q_values = self.target_critic(
            torch.cat(
                [next_states, self.target_actor(next_states)], dim=1))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(
            F.mse_loss(
                # MSE损失函数
                self.critic(torch.cat([states, actions], dim=1)),
                q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 注意critic搭配actor使用
        actor_loss = -torch.mean(
            self.critic(
                # 策略网络就是为了使得Q值最大化
                torch.cat([states, self.actor(states)], dim=1)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络



if __name__ == '__main__': 
    tau = 0.005 # 软更新参数
    sigma = 0.01 # 高斯噪声标准差 
    buffer_size = 10000
    minial_size = 1000
    batch_size = 64
    replay_buffer = ReplayBuffer(buffer_size)
    env = gym.make(ENV_ID)
    env.reset()
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    agent = DDPG(state_dim, action_dim, state_dim + action_dim, HIDDEN_SIZE, False, action_bound, sigma, LEARNING_RATE, LEARNING_RATE, tau, GAMMA, DEVICE)
    num_episodes = 100
    start_time = time.time()
    return_list = []
    for i in range(EPOCH_NUM):
        for i_episode in range(num_episodes):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state                
                
                episode_return += reward

                if replay_buffer.size() > minial_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)   # 注意这里不需要等一个回合结束再开始训练（跟DQN一样，跟PG不同）

            return_list.append(episode_return)
            
            progress = i_episode + i * num_episodes + 1
            if progress % 10 == 0:
                print(f'progress: {progress} / {num_episodes * EPOCH_NUM} | elapse: {time.time() - start_time:.3f}s | return: {np.mean(return_list[-10:])}')


    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(ENV_ID))
    plt.show()