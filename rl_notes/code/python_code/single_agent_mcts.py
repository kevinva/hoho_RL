import random

from mcts import Node
from mcts import MCTS

class SingleAgentNode(Node):
    def __init__(
        self,
        mdp,
        parent,
        state,
        qfunction,
        bandit,
        reward=0.0,
        action=None,
    ):
        super().__init__(mdp, parent, state, qfunction, bandit, reward, action)

        print(f"create state = {state}, parent = {parent.state if parent is not None else None}, action = {action}, reward = {reward}")
        # A dictionary from actions to a set of node-probability pairs
        self.children = {}

    """ Return true if and only if all child actions have been expanded """

    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state)
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    """ Select a node that is not fully expanded """

    def select(self):
        # 该节点要完全展开完，才会选其子节点
        if not self.is_fully_expanded() or self.mdp.is_terminal(self.state):
            print(f"select myself: state = {self.state}")
            return self
        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.state, actions, self.qfunction)
            print(f"select child: action = {action}， from state = {self.state}")
            return self.get_outcome_child(action).select()

    """ Expand a node if it is not a terminal node """

    def expand(self):
        print("expand!!!")
        if not self.mdp.is_terminal(self.state):
            # Randomly select an unexpanded action to expand
            # 其实就是选一个从来没执行过的动作来expand，即每次只展开一个子节点
            actions = self.mdp.get_actions(self.state) - self.children.keys()
            action = random.choice(list(actions))

            self.children[action] = []
            return self.get_outcome_child(action)
        return self

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child):
        action = child.action   # 触发这个child节点的action

        Node.visits[self.state] = Node.visits[self.state] + 1
        Node.visits[(self.state, action)] = Node.visits[(self.state, action)] + 1

        q_value = self.qfunction.get_q_value(self.state, action)
        delta = (1 / (Node.visits[(self.state, action)])) * (
            reward - self.qfunction.get_q_value(self.state, action)
        )
        self.qfunction.update(self.state, action, delta)

        if self.parent != None:
            self.parent.back_propagate(self.reward + reward, self)

    """ Simulate the outcome of an action, and return the child node """

    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities
        (next_state, reward) = self.mdp.execute(self.state, action)

        # Find the corresponding state and return if this already exists
        for (child, _) in self.children[action]:
            if next_state == child.state:
                return child

        # This outcome has not occured from this state-action pair previously
        new_child = SingleAgentNode(
            self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
        )

        # Find the probability of this outcome (only possible for model-based) for visualising tree
        probability = 0.0
        for (outcome, probability) in self.mdp.get_transitions(self.state, action):
            if outcome == next_state:
                self.children[action] += [(new_child, probability)]
                return new_child

class SingleAgentMCTS(MCTS):
    def create_root_node(self):
        return SingleAgentNode(
            self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit
        )


if __name__ == "__main__":

    import time
    from gridworld import GridWorld
    from graph_visualisation import GraphVisualisation
    from qtable import QTable
    from single_agent_mcts import SingleAgentMCTS
    from multi_armed_bandit.ucb import UpperConfidenceBounds

    gridWorld = GridWorld()
    qfunction = QTable()
    mcts_algorithm = SingleAgentMCTS(gridWorld, qfunction, UpperConfidenceBounds()).mcts(timeout = 0.05)
    gv = GraphVisualisation(max_level = 6)
    graph = gv.single_agent_mcts_to_graph(mcts_algorithm, filename = f"./outputs/mcts_{int(time.time())}")
    graph.view()

    gridWorld.visualise_q_function(qfunction)
    
    policy = qfunction.extract_policy(gridWorld)
    gridWorld.visualise_policy(policy)
