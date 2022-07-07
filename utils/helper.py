import time
import numpy as np
import torch


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def train_on_policy_agent(env, agent, num_epoch, num_episodes):
    return_list = []
    for i in range(num_epoch):
        start_time = time.time()
        for i_episode in range(num_episodes):
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
            progress = (i_episode + 1) + i * num_episodes
            total = num_epoch * num_episodes
            if progress % 10 == 0:
                print(f'progress={progress} / {total} | elapse={time.time() - start_time} | average return={np.mean(return_list[-10:])}')
    return return_list


def train_off_policy_agent(env, agent, num_epoch, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(num_epoch):
        start_time = time.time()
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
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)
            return_list.append(episode_return)
            progress = (i_episode + 1) + i * num_episodes
            total = num_epoch * num_episodes
            if progress % 10 == 0:
                print(f'progress={progress} / {total} | elapse={time.time() - start_time} | average return={np.mean(return_list[-10:])}')
    return return_list