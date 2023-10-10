import math
import random
from multi_armed_bandit import MultiArmedBandit


class UpperConfidenceBounds(MultiArmedBandit):
    def __init__(self):
        self.total = 0
        # number of times each action has been chosen
        self.times_selected = {}

    def select(self, state, actions, qfunction):

        # First execute each action one time
        for action in actions:
            if action not in self.times_selected.keys():
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_actions = []
        max_value = float("-inf")
        for action in actions:
            value = qfunction.get_q_value(state, action) + math.sqrt(
                (2 * math.log(self.total)) / self.times_selected[action]
            )
            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(max_actions)

        self.times_selected[result] = self.times_selected[result] + 1
        self.total += 1

        return result
    


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from multi_armed_bandit import run_bandit

    from epsilon_greedy import EpsilonGreedy
    from epsilon_decreasing import EpsilonDecreasing
    from softmax import Softmax

    drift = False
    epsilon_greedy = run_bandit(EpsilonGreedy(epsilon=0.1), drift=drift)
    epsilon_decreasing = run_bandit(EpsilonDecreasing(alpha=0.99), drift=drift)
    softmax = run_bandit(Softmax(tau=1.0), drift=drift)
    ucb = run_bandit(UpperConfidenceBounds(), drift=drift)

    epsilon_greedy = np.array(epsilon_greedy)
    epsilon_greedy_avg = np.mean(epsilon_greedy, axis=0)
    epsilon_decreasing = np.array(epsilon_decreasing)
    epsilon_decreasing_avg = np.mean(epsilon_decreasing, axis=0)
    softmax = np.array(softmax)
    softmax_avg = np.mean(softmax, axis=0)
    ucb = np.array(ucb)
    ucb_avg = np.mean(ucb, axis=0)

    plt.figure(figsize=(10, 5))
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Steps")
    plt.plot(epsilon_greedy_avg, label="epsilon greedy")
    plt.plot(epsilon_decreasing_avg, label="epsilon decreasing")
    plt.plot(softmax_avg, label="softmax")
    plt.plot(ucb_avg, label="ucb")
    plt.legend()
    plt.savefig("../outputs/ucb.png")
