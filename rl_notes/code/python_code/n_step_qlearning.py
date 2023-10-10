from n_step_reinforcement_learner import NStepReinforcementLearner

class NStepQLearning(NStepReinforcementLearner):
    def state_value(self, state, action):
        (_, max_q_value) = self.qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value


if __name__ == "__main__":
    from gridworld import GridWorld
    from qtable import QTable
    from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
    
    gridworld = GridWorld()
    qfunction = QTable()
    NStepQLearning(
        gridworld,
        EpsilonGreedy(epsilon=0.1),
        qfunction,
        n=5,
        alpha=0.1,
    ).execute(100)

    gridworld.visualise_q_function(qfunction)