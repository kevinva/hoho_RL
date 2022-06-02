import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor(np.array([action_list[i]]), dtype=torch.int64).view(-1, 1).to(self.device)
            # print(f'action: {action}')
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


if __name__ == '__main__':

    import gym
    import time
    import matplotlib.pyplot as plt

    learning_rate = 1e-3
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

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

            agent.update(transition_dict)  # 注意这里于DQN的不同：需要一个完整的片段（即游戏结束）才能开始训练

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
