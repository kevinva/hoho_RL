import torch
import torch.multiprocessing as mp
import random
import gym
import numpy as np
from models.dqn import Qnet
from utils.replay_buffer import *
from models.dqn_mp import DQN, DQNActionSelector, DQNAgent

ENVIRONMENT_NAME = 'CartPole-v1'
LEARNING_RATE = 2e-3
NUM_EPOCH = 10
NUM_EPISODES = 100
HIDDEN_DIM = 128
GAMMA = 0.98
EPSILON = 0.01
TARGET_UPDATE_INTERVAL = 10
BUFFER_SIZE = 10000
MINIMAL_SIZE = 500
BATCH_SIZE = 64
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EpisodeEnd(object):

    def __init__(self, episode_i, epoch_i, episode_return):
        super(EpisodeEnd, self).__init__()

        self.episode_i = episode_i
        self.epoch_i = epoch_i
        self.episode_return = episode_return


def play(net, exp_queue):
    env = gym.make(ENVIRONMENT_NAME)
    random.seed(0)
    selector = DQNActionSelector(net, env.action_space.n, EPSILON, DEVICE)

    for i in range(10):
        for episode in range(NUM_EPISODES):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = selector.take_action(state)
                next_state, reward, done, _ = env.step(action)
                exp_queue.put((state, action, reward, next_state, done))
                state = next_state
                episode_return += reward
            
            episode_end = EpisodeEnd(episode, i, episode_return)
            exp_queue.put(episode_end)


def train(net, exp_queue):
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    agent = DQNAgent(net, LEARNING_RATE, GAMMA, TARGET_UPDATE_INTERVAL, DEVICE)
    return_list = []
    smooth_return_list = []
    trian_finish = False
    while not trian_finish:
        while exp_queue.qsize() > 0:
            print(f'queue size: {exp_queue.qsize()}')
            
            exp = exp_queue.get()
            if isinstance(exp, EpisodeEnd):
                return_list.append(exp.episode_return)

                if len(smooth_return_list) == 0:
                    smooth_return_list.append(exp.episode_return)
                else:
                    smooth_return_list.append(smooth_return_list[-1] * 0.9 + exp.episode_return * 0.1)
                    
                if (exp.episode_i +  exp.epoch_i * NUM_EPISODES + 1) % 10 == 0:
                    print(f'{exp.episode_i +  exp.epoch_i * NUM_EPISODES + 1} / {NUM_EPISODES * NUM_EPOCH}')

                trian_finish = (exp.episode_i +  exp.epoch_i * NUM_EPISODES + 1 == NUM_EPISODES * NUM_EPOCH)    
            else:
                replay_buffer.add(exp)
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




if __name__=='__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = gym.make(ENVIRONMENT_NAME)
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    qnet = Qnet(state_dim, HIDDEN_DIM, action_dim)

    mp.set_start_method('spawn')
    exp_queue = mp.Queue(maxsize=100)
    play_proc = mp.Process(target=play, args=(qnet, exp_queue))
    play_proc.start()
    play_proc.join()

    train(qnet, exp_queue)

