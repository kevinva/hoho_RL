from math import gamma
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import os, sys
root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir_path)

from utils.const import *
from utils.replay_buffer import *
from utils.helper import *


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = torch.zeros(td_target.size())
        for idx, done in enumerate(dones):
            if not (done == 1):
                td_delta[idx] = td_target[idx] - self.critic(states[idx].unsqueeze(0))
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == '__main__':
    num_episodes = 200
    hidden_dim = 128
    lmbda = 0.95
    eps = 0.2
    actor_lr = 1e-3
    critic_lr = 1e-2
    distrill_lr = 1e-3
    
    env_name = 'MountainCar-v0'
    # env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    net_distrill = MountainCarNetDistillery(state_dim).to(DEVICE)
    distrillor = MountainCarDistrillor(distrill_lr, net_distrill)
    env = NetworkDistillationRewardWrapper(env, net_distrill.extra_reward, DEVICE)
    env.reset(seed=0)
    torch.manual_seed(0)
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, EPOCH_NUM, eps, GAMMA, DEVICE)
    # return_list = train_on_policy_agent(env, agent, EPOCH_NUM, num_episodes)

    return_list = []
    for i in range(EPOCH_NUM):
        for i_episode in range(num_episodes):
            start_time = time.time()
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            distrillor.update(transition_dict['states'], DEVICE)

            progress = (i_episode + 1) + i * num_episodes
            total = EPOCH_NUM * num_episodes
            if progress % 10 == 0:
                print(f'progress={progress} / {total} | elapse={time.time() - start_time} | average return={np.mean(return_list[-10:])}')

    episode_list = list(range(len(return_list)))
    plt.plot(episode_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'PPO on {env_name}')
    plt.show()