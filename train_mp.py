import torch
import torch.multiprocessing as mp
import random
import time
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from utils.replay_buffer import *
from models.dqn_mp import Qnet, DQNActionSelector, DQNAgent

ENVIRONMENT_NAME = 'CartPole-v1'
LEARNING_RATE = 1e-3
NUM_EPOCH = 10
NUM_EPISODES = 100
HIDDEN_DIM = 128
GAMMA = 0.98
EPSILON = 0.01
TARGET_UPDATE_INTERVAL = 10
BUFFER_SIZE = 10000
MINIMAL_SIZE = 500
BATCH_SIZE = 64
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # hoho: 使用多进程时，不知为啥在GPU上会模型不能共享更新
DEVICE = torch.device('cpu')

class EpisodeEnd(object):

    def __init__(self, episode_i, epoch_i, episode_return):
        super(EpisodeEnd, self).__init__()

        self.episode_i = episode_i
        self.epoch_i = epoch_i
        self.episode_return = episode_return


def play(net, exp_queue):
    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=0)
    selector = DQNActionSelector(net, env.action_space.n, EPSILON, DEVICE)

    print(f'sub pid: {os.getpid()}')

    for i in range(NUM_EPOCH):
        for episode in range(NUM_EPISODES):
            episode_return = 0
            state = env.reset()
            done = False

            while not done:
                action = selector.take_action(state)
                next_state, reward, done, _ = env.step(action)
                exp_queue.put((state, action, reward, next_state, done))
                # print(f'play! pid: {os.getpid()}, queue size: {exp_queue.qsize()}')

                state = next_state
                episode_return += reward
            
            episode_end = EpisodeEnd(episode, i, episode_return)
            exp_queue.put(episode_end)

            # print(f'episode finish! pid: {os.getpid()}, queue size: {exp_queue.qsize()}')

    print(f'done! pid: {os.getpid()}')


def train(net, target_net, exp_queue):
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    agent = DQNAgent(net, target_net, LEARNING_RATE, GAMMA, TARGET_UPDATE_INTERVAL, DEVICE)
    return_list = []
    smooth_return_list = []
    trian_finish = False
    start_time = time.time()

    print(f'pid: {os.getpid()}')

    while not trian_finish:
        while exp_queue.qsize() > 0:
            # print(f'pid: {os.getpid()}, queue size: {exp_queue.qsize()}')
            
            exp = exp_queue.get()
            if isinstance(exp, EpisodeEnd):
                return_list.append(exp.episode_return)

                if len(smooth_return_list) == 0:
                    smooth_return_list.append(exp.episode_return)
                else:
                    smooth_return_list.append(smooth_return_list[-1] * 0.9 + exp.episode_return * 0.1)
                
                progress = exp.episode_i +  exp.epoch_i * NUM_EPISODES + 1
                if progress % 10 == 0:
                    print(f'progress: {progress} / {NUM_EPISODES * NUM_EPOCH}, elapse: {time.time() - start_time}, return: {np.mean(return_list[-10:])}')

                trian_finish = (progress == NUM_EPISODES * NUM_EPOCH)    
            else:
                state, action, reward, next_state, done = exp
                replay_buffer.add(state, action, reward, next_state, done)
                if replay_buffer.size() > MINIMAL_SIZE:
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)
                    transition_dict = {
                        'states': batch_state,
                        'actions': batch_action,
                        'next_states': batch_next_state,
                        'rewards': batch_reward,
                        'dones': batch_done
                    }
                    agent.update(transition_dict)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.plot(episodes_list, smooth_return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(ENVIRONMENT_NAME))
    plt.savefig(f'.\output\dqn_{int(time.time())}.png')


if __name__=='__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    mp.set_start_method('spawn')

    env = gym.make(ENVIRONMENT_NAME)
    env.reset(seed=0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    qnet = Qnet(state_dim, HIDDEN_DIM, action_dim).to(DEVICE)
    qnet.share_memory()
    target_qnet = Qnet(state_dim, HIDDEN_DIM, action_dim).to(DEVICE)

    exp_queue = mp.Queue(maxsize=20)

    play_proc = mp.Process(target=play, args=(qnet, exp_queue))
    play_proc.start()

    train(qnet, target_qnet, exp_queue)

    play_proc.join()
