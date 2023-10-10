import random
from .multi_armed_bandit import MultiArmedBandit


class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def reset(self):
        pass

    def select(self, state, actions, qfunction):
        # Select a random action with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(actions)
        (arg_max_q, _) = qfunction.get_max_q(state, actions)
        return arg_max_q
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from multi_armed_bandit import run_bandit
    

    drift = False
    epsilon000 = run_bandit(EpsilonGreedy(epsilon=0.00), drift=drift)
    epsilon005 = run_bandit(EpsilonGreedy(epsilon=0.05), drift=drift)
    epsilon01 = run_bandit(EpsilonGreedy(epsilon=0.1), drift=drift)
    epsilon02 = run_bandit(EpsilonGreedy(epsilon=0.2), drift=drift)
    epsilon04 = run_bandit(EpsilonGreedy(epsilon=0.4), drift=drift)
    epsilon08 = run_bandit(EpsilonGreedy(epsilon=0.8), drift=drift)
    epsilon10 = run_bandit(EpsilonGreedy(epsilon=1.0), drift=drift)

    epsilon000 = np.array(epsilon000)
    epsilon000_avg = np.mean(epsilon000, axis=0)
    epsilon005 = np.array(epsilon005)
    epsilon005_avg = np.mean(epsilon005, axis=0)
    epsilon01 = np.array(epsilon01)
    epsilon01_avg = np.mean(epsilon01, axis=0)
    epsilon02 = np.array(epsilon02)
    epsilon02_avg = np.mean(epsilon02, axis=0)
    epsilon04 = np.array(epsilon04)
    epsilon04_avg = np.mean(epsilon04, axis=0)
    epsilon08 = np.array(epsilon08)
    epsilon08_avg = np.mean(epsilon08, axis=0)
    epsilon10 = np.array(epsilon10)
    epsilon10_avg = np.mean(epsilon10, axis=0)


    plt.figure(figsize=(10, 5))
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Steps")
    plt.plot(epsilon000_avg, label="epsilon = 0.0")
    plt.plot(epsilon005_avg, label="epsilon = 0.05")
    plt.plot(epsilon01_avg, label="epsilon = 0.1")
    plt.plot(epsilon02_avg, label="epsilon = 0.2")
    plt.plot(epsilon04_avg, label="epsilon = 0.4")
    plt.plot(epsilon08_avg, label="epsilon = 0.8")
    plt.plot(epsilon10_avg, label="epsilon = 1.0")
    plt.legend()
    # plt.show()
    plt.savefig("../outputs/epsilon_greedy.png")
