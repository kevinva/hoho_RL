import math
import random
from multi_armed_bandit import MultiArmedBandit


class Softmax(MultiArmedBandit):
    def __init__(self, tau=1.0):
        self.tau = tau

    def reset(self):
        pass

    def select(self, state, actions, qfunction):

        # Boltzman distribution
        # calculate the denominator for the softmax strategy
        total = 0.0
        for action in actions:
            total += math.exp(qfunction.get_q_value(state, action) / self.tau)

        rand = random.random()
        cumulative_probability = 0.0
        result = None
        for action in actions:
            probability = (
                math.exp(qfunction.get_q_value(state, action) / self.tau) / total
            )
            if cumulative_probability <= rand <= cumulative_probability + probability:
                result = action
            cumulative_probability += probability

        return result
    


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from multi_armed_bandit import run_bandit
    
    drift = True
    tau10 = run_bandit(Softmax(tau=1.0), drift=drift)
    tau11 = run_bandit(Softmax(tau=1.1), drift=drift)
    tau15 = run_bandit(Softmax(tau=1.5), drift=drift)
    tau20 = run_bandit(Softmax(tau=2.0), drift=drift)

    tau10 = np.array(tau10)
    tau10_avg = np.mean(tau10, axis=0)
    tau11 = np.array(tau11)
    tau11_avg = np.mean(tau11, axis=0)
    tau15 = np.array(tau15)
    tau15_avg = np.mean(tau15, axis=0)
    tau20 = np.array(tau20)
    tau20_avg = np.mean(tau20, axis=0)

    plt.figure(figsize=(10, 5))
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Steps")
    plt.plot(tau10_avg, label="tau = 1.0")
    plt.plot(tau11_avg, label="tau = 1.1")
    plt.plot(tau15_avg, label="tau = 1.5")
    plt.plot(tau20_avg, label="tau = 2.0")
    plt.legend()
    plt.savefig("../outputs/softmax_drift.png")
