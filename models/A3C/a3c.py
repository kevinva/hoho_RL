import random
from tracemalloc import start
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal, Categorical
# from torch.multiprocessing import Process
import matplotlib.pyplot as plt
import time
from worker import Worker
from net import *
from const import *
import gym


class A3C():
    def __init__(self, env, continuous, state_size, action_size):
        self.max_episode = GLOBAL_MAX_EPISODE
        self.global_episode = mp.Value('i', 0)  # 进程之间共享的变量，'i'为整型
        self.global_epi_rew = mp.Value('d', 0)  # 'd'为浮点型
        self.rew_queue = mp.Queue()
        self.worker_num = mp.cpu_count()

        # define the global networks
        self.global_valueNet = ValueNetwork(state_size,1).to(DEVICE)
        # global 的网络参数放入 shared memory，以便复制给各个进程中的 worker网络
        self.global_valueNet.share_memory()

        if continuous:
            self.global_policyNet = ActorContinous(state_size, action_size).to(DEVICE)
        else:
            self.global_policyNet = ActorDiscrete(state_size, action_size).to(DEVICE)
        self.global_policyNet.share_memory()

        # global optimizer
        self.global_optimizer_policy = optim.Adam(self.global_policyNet.parameters(), lr=LR)
        self.global_optimizer_value = optim.Adam(self.global_valueNet.parameters(), lr=LR)

        print(f'worker num: {self.worker_num}')

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

# import time

# def myFunc(queue):
#     time.sleep(3)
#     queue.put(3)
#     print('myFunc finish!')

def train_agent_for_env(env_name, continuous):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    agent = A3C(env, continuous, state_dim, action_dim)
    scores = agent.train_worker()

    return agent, scores


if __name__ == '__main__':
    # myqueue = mp.Queue()
    # myProcess = mp.Process(target=myFunc, args=(myqueue,))
    # myProcess.start()
    # print('process start')
    # val = myqueue.get()
    # myProcess.join()
    # print('val=', val)

    start_time = time.time()

    agent, scores = train_agent_for_env('CartPole-v1', False)

    print(f'time spend: {time.time() - start_time :.3f}')

    plt.plot(range(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()