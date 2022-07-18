import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import collections


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
            progress = (i_episode + 1) + i * num_episodes
            total = num_epoch * num_episodes
            if progress % 10 == 0:
                print(f'progress={progress} / {total} | elapse={time.time() - start_time} | average return={np.mean(return_list[-10:])}')
    return return_list


def train_off_policy_agent(env, agent, num_epoch, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(num_epoch):
        for i_episode in range(num_episodes):
            start_time = time.time()
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



class PseudoCountRewardWrapper(gym.Wrapper):
    def __init__(self, env, hash_function = lambda o: o,
                 reward_scale: float = 1.0):
        super(PseudoCountRewardWrapper, self).__init__(env)
        self.hash_function = hash_function
        self.reward_scale = reward_scale
        self.counts = collections.Counter()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extra_reward = self._count_observation(obs)
        return obs, reward + self.reward_scale * extra_reward, \
               done, info

    def _count_observation(self, obs) -> float:
        """
        Increments observation counter and returns pseudo-count reward
        :param obs: observation
        :return: extra reward
        """
        h = self.hash_function(obs)
        self.counts[h] += 1
        return np.sqrt(1/self.counts[h])

def counts_hash(obs):
    r = obs.tolist()
    return tuple(map(lambda v: round(v, 3), r))


class MountainCarNetDistillery(nn.Module):
    def __init__(self, obs_size: int, hid_size: int = 128):
        super(MountainCarNetDistillery, self).__init__()

        self.ref_net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        self.ref_net.train(False)

        self.trn_net = nn.Sequential(
            nn.Linear(obs_size, 1),
        )

    def forward(self, x):
        return self.ref_net(x), self.trn_net(x)

    def extra_reward(self, obs, device):
        r1, r2 = self.forward(torch.FloatTensor([obs]).to(device))
        return (r1 - r2).abs().detach().cpu().numpy()[0][0]

    def loss(self, obs_t):
        r1_t, r2_t = self.forward(obs_t)
        return F.mse_loss(r2_t, r1_t).mean()


class NetworkDistillationRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_callable, device, reward_scale: float = 1.0, sum_rewards: bool = True):
        super(NetworkDistillationRewardWrapper, self).__init__(env)
        self.reward_scale = reward_scale
        self.reward_callable = reward_callable
        self.sum_rewards = sum_rewards
        self.device = device

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        extra_reward = self.reward_callable(obs, self.device)
        if self.sum_rewards:
            res_rewards = reward + self.reward_scale * extra_reward
        else:
            res_rewards = np.array([reward, extra_reward * self.reward_scale])
        return obs, res_rewards, done, info


class MountainCarDistrillor:

    def __init__(self, lr, distrill_net: MountainCarNetDistillery):
        self.distrill_net = distrill_net
        self.optim = optim.Adam(distrill_net.trn_net.parameters(), lr=lr)

    def update(self, states, device):
        batch_states = torch.FloatTensor(states).to(device)
        self.optim.zero_grad()
        loss = self.distrill_net.loss(batch_states)
        loss.backward()
        self.optim.step()

        # print(f'distrill loss={loss.item()}')

        return loss.item()
