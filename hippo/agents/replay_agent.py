import copy


class ReplayAgent:
    def __init__(self, actions):
        self.actions = list(copy.copy(actions))

    def reset(self):
        pass

    def act(self, obs, reward):
        return self.actions.pop(0), {}
