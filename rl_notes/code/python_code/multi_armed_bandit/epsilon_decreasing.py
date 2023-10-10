from multi_armed_bandit import MultiArmedBandit
from epsilon_greedy import EpsilonGreedy

# 前期exploration，后期exploitation，通常是一个好的策略
class EpsilonDecreasing(MultiArmedBandit):
    def __init__(self, epsilon=0.2, alpha=0.999):
        self.epsilon_greedy_bandit = EpsilonGreedy(epsilon)
        self.initial_epsilon = epsilon
        self.alpha = alpha

    def reset(self):
        self.epsilon_greedy_bandit = EpsilonGreedy(self.initial_epsilon)

    def select(self, state, actions, qfunction):
        result = self.epsilon_greedy_bandit.select(state, actions, qfunction)
        self.epsilon_greedy_bandit.epsilon *= self.alpha
        return result
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from multi_armed_bandit import run_bandit

    drift = False
    alpha09 = run_bandit(EpsilonDecreasing(alpha=0.9), drift=drift)
    alpha099 = run_bandit(EpsilonDecreasing(alpha=0.99), drift=drift)
    alpha0999 = run_bandit(EpsilonDecreasing(alpha=0.999), drift=drift)
    alpha1 = run_bandit(EpsilonDecreasing(alpha=1.0), drift=drift)

    alpha09 = np.array(alpha09)
    alpha09_avg = np.mean(alpha09, axis=0)
    alpha099 = np.array(alpha099)
    alpha099_avg = np.mean(alpha099, axis=0)
    alpha0999 = np.array(alpha0999)
    alpha0999_avg = np.mean(alpha0999, axis=0)
    alpha1 = np.array(alpha1)
    alpha1_avg = np.mean(alpha1, axis=0)

    plt.figure(figsize=(10, 5))
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Steps")
    plt.plot(alpha09_avg, label="alpha = 0.9")
    plt.plot(alpha099_avg, label="alpha = 0.99")
    plt.plot(alpha0999_avg, label="alpha = 0.999")
    plt.plot(alpha1_avg, label="alpha = 1.0")
    plt.legend()
    plt.savefig("../outputs/epsilon_decreasing.png")


