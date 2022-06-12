import gym
import torch
import torch.nn.functional as F
from torch import nn, optim
import time
import numpy as np
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络

        # 策略网络优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
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

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差  # critic网络输出V值，这里td_delta可以看作优势函数（A=Q-V）
        log_probs = torch.log(self.actor(states).gather(1, actions))

        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # print(f'actions: {actions}')
        # print(f'log_probs: {log_probs}')
        # print(f'actor_loss: {actor_loss}')

        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数


if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

    start_time = time.time()

    return_list = []
    average_return_list = []
    for i in range(10):
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
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            if len(average_return_list) == 0:
                average_return_list.append(episode_return)
            else:
                average_return_list.append(average_return_list[-1] * 0.8 + episode_return * 0.2)

            agent.update(transition_dict)  # 注意这里于DQN的不同：需要一个完整的片段（即游戏结束）才能开始训练(on-policy)

            progress = i_episode + i * num_episodes + 1
            if progress % 10 == 0:
                print(f'progress: {progress} / {num_episodes * 10}, elapse: {time.time() - start_time}, return: {np.mean(return_list[-10:])}')
   
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.plot(episodes_list, average_return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()