
import torch
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
from torch import nn
import numpy as np
from const import DEVICE

class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)

        return value


class ActorDiscrete(nn.Module):
    """
    用于离散动作空间的策略网络
    """
    def __init__(self, state_size, action_size):
        super(ActorDiscrete, self).__init__()
        # self.seed = torch.manual_seed(0)  # hoho: 使用多进程时，在子进程里不能写这句！！！
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(x), dim=1)
        return out

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(DEVICE)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()

        # return action for current state, and the corresponding probability
        # hoho: probs[:, action.item()]为啥probs有两个维度
        return action.item(), probs[:, action.item()].item()


class ActorContinous(nn.Module):
    """
    用于连续动作空间的策略网络
    """
    def __init__(self, state_size, action_size):
        super(ActorContinous, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu_head = nn.Linear(128, action_size)
        self.sigma_head = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)

    def take_action(self,state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(DEVICE)
        with torch.no_grad():
            (mu, sigma) = self.forward(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.numpy()[0], action_log_prob.numpy()[0]