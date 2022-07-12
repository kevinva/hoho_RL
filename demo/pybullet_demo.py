import gym
import pybullet_envs


ENV_ID = 'MinitaurBulletEnv-v0'
RENDER = True


if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV_ID)
    spec.kwargs['render'] = RENDER
    print(spec)
    env = gym.make(ENV_ID)

    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')
    print(env)
    print(env.reset())
    input('Press any key to exit\n')
    env.close()