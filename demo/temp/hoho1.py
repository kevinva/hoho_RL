import torch
import gym



e = gym.make('MountainCar-v0')
print(e.reset())
print(e.observation_space.shape[0])
print(e.action_space.n)
print(e.step(0))
print(e.step(0))