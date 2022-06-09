import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from worker import Worker

GAMMA = 0.9
LR = 1e-4
GLOBAL_MAX_EPISODE = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(x), dim=1)
        return out

    def take_action(self, state):
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
        with torch.no_grad():
            (mu, sigma) = self.forward(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.numpy()[0], action_log_prob.numpy()[0]


class A3C():
    def __init__(self, env, continuous, state_size, action_size):
        self.max_episode = GLOBAL_MAX_EPISODE
        self.global_episode = mp.Value('i', 0)  # 进程之间共享的变量，'i'为整型
        self.global_epi_rew = mp.Value('d', 0)  # 'd'为浮点型
        self.rew_queue = mp.Queue()
        self.worker_num = mp.cpu_count()

        # define the global networks
        self.global_valueNet = ValueNetwork(state_size,1).to(device)
        # global 的网络参数放入 shared memory，以便复制给各个进程中的 worker网络
        self.global_valueNet.share_memory()

        if continuous:
            self.global_policyNet = ActorContinous(state_size, action_size).to(device)
        else:
            self.global_policyNet = ActorDiscrete(state_size, action_size).to(device)
        self.global_policyNet.share_memory()

        # global optimizer
        self.global_optimizer_policy = optim.Adam(self.global_policyNet.parameters(), lr=LR)
        self.global_optimizer_value = optim.Adam(self.global_valueNet.parameters(), lr=LR)

        # define the workers
        self.workers = [Worker(env,
                               continuous,
                               state_size,
                               action_size,
                               i,
                               self.global_valueNet,
                               self.global_optimizer_value,
                               self.global_policyNet,
                               self.global_optimizer_policy,
                               self.global_episode,
                               self.global_epi_rew,
                               self.rew_queue,
                               self.max_episode,
                               GAMMA)

                       for i in range(self.worker_num)]

    def train_worker(self):
        scores = []
        [w.start() for w in self.workers]
        while True:
            r = self.rew_queue.get()  # 如果队列里面没有值，会一直等待队列直到有值，即可取出来
            if r is not None:
                scores.append(r)
            else:
                break
        [w.join() for w in self.workers]

        return scores

    def save_model(self):
        torch.save(self.global_valueNet.state_dict(), "a3c_value_model.pth")
        torch.save(self.global_policyNet.state_dict(), "a3c_policy_model.pth")

import time

def myFunc(queue):
    time.sleep(3)
    queue.put(3)
    print('myFunc finish!')


if __name__ == '__main__':
    myqueue = mp.Queue()
    myProcess = mp.Process(target=myFunc, args=(myqueue,))
    myProcess.start()
    print('process start')
    val = myqueue.get()
    myProcess.join()
    print('val=', val)