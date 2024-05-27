import numpy as np

from hippo.free_energy import expected_free_energy


class TreeNode:
    def __init__(self, action, data):
        self.action = action
        self.data = data

    def actions(self, transition, filter_actions=True):
        actions = set(np.arange(transition.shape[0]))

        # Do not turn left after turn right!
        if filter_actions:
            if self.action == 0:
                actions.remove(1)
            elif self.action == 1:
                actions.remove(0)

        return actions


def construct_tree(
    agent, state_belief, constraint, filter_actions=True, epistemic=True
):
    start_node = TreeNode(
        np.argmax(state_belief), {"nefe": 0, "qs": state_belief, "step": 0}
    )
    # Construct a list of policies by rolling out the tree
    policies = [[]]
    for t in range(agent.policy_len):
        result = []
        for policy in policies:
            if len(policy) == 0:
                last_node = start_node
            else:
                last_node = policy[-1]

            result += expand_tree_one_step(
                agent, policy, last_node, constraint, filter_actions, epistemic
            )
        policies = result
    return policies


def expand_tree_one_step(
    agent, policy, last_node, constraint, filter_actions=True, use_epistemic=True
):
    new_policies = []
    actions = last_node.actions(agent.T, filter_actions)
    for action in actions:
        efe, data = expected_free_energy(
            agent, last_node, action, constraint, use_epistemic
        )

        # Add node to the policy
        node = TreeNode(action, data)
        new_policies.append(policy + [node])
    return new_policies
