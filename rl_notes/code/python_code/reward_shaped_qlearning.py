from model_free_reinforcement_learner import ModelFreeReinforcementLearner
from qlearning import QLearning


class RewardShapedQLearning(QLearning):
    def __init__(self, mdp, bandit, potential, qfunction, alpha=0.1):
        super().__init__(mdp, bandit, qfunction=qfunction, alpha=alpha)
        self.potential = potential

    def get_delta(self, reward, q_value, state, next_state, next_action):
        next_state_value = self.state_value(next_state, next_action)
        state_potential = self.potential.get_potential(state)
        next_state_potential = self.potential.get_potential(next_state)
        potential = self.mdp.discount_factor * next_state_potential - state_potential
        delta = reward + potential + self.mdp.discount_factor * next_state_value - q_value
        return delta


if __name__ == "__main__":
    from gridworld import GridWorld
    from qtable import QTable
    from qlearning import QLearning
    from gridworld_potential_function import GridWorldPotentialFunction
    from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

    mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
    qfunction = QTable()
    potential = GridWorldPotentialFunction(mdp)
    # RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute()
    QLearning(mdp, EpsilonGreedy(), qfunction).execute()  # 不使用reward shaping
    policy = qfunction.extract_policy(mdp)
    mdp.visualise_q_function(qfunction)
    mdp.visualise_policy(policy)
    reward_shaped_rewards = mdp.get_rewards()
