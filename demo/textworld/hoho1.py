import textworld
import textworld.gym
from textworld.text_utils import extract_vocab_from_gamefile
from textworld.gym.spaces.text_spaces import Word

from textworld import EnvInfos
import gym

vocab = list(extract_vocab_from_gamefile('hoho1.z8'))
action_space = Word(max_length=8, vocab=vocab)
observation_space = Word(max_length=len(vocab), vocab=vocab)
print(action_space)
print(action_space.tokenize('go east'))
act = action_space.sample()
print(act)
print([action_space.id2w[a] for a in act])


request_infos = EnvInfos(inventory=True, intermediate_reward=True, admissible_commands=True)
env_id = textworld.gym.register_game('hoho1.z8', 
                                     request_infos=request_infos, 
                                     action_space=action_space,
                                     observation_space=observation_space)
env = gym.make(env_id)
print(env)
print(env.observation_space)
print(env.action_space)
r = env.reset()
# print(r)
r1 = env.step('go east')
# print(r1)