from gridworld import *
from tabular_value_function import *
from qtable import *


class ValueIteration:
    def __init__(self, mdp, values):
        self.mdp = mdp
        self.values = values

    def value_iteration(self, max_iterations=100, theta=0.001):

        for i in range(max_iterations):
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                qtable = QTable()
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = 0.0
                    for (new_state, probability) in self.mdp.get_transitions(
                        state, action
                    ):
                        reward = self.mdp.get_reward(state, action, new_state)
                        print(f"state = {state}, action = {action}, new_state = {new_state}, reward = {reward}")

                        new_value += probability * (
                            reward
                            + (
                                self.mdp.get_discount_factor()
                                * self.values.get_value(new_state)
                            )
                        )

                    qtable.update(state, action, new_value)

                # V(s) = max_a Q(sa)
                (_, max_q) = qtable.get_max_q(state, self.mdp.get_actions(state))
                delta = max(delta, abs(self.values.get_value(state) - max_q))
                new_values.update(state, max_q)

            self.values.merge(new_values)

            # Terminate if the value function has converged
            if delta < theta:
                return i



if __name__ == "__main__":
    from gridworld import GridWorld
    from tabular_value_function import TabularValueFunction

    gridworld = GridWorld()
    values = TabularValueFunction()
    ValueIteration(gridworld, values).value_iteration(max_iterations = 100)
    gridworld.visualise_value_function(values, "Value function after 100 iterations")

    policy = values.extract_policy(gridworld)
    gridworld.visualise_policy(policy, "Policy after 100 iterations")

