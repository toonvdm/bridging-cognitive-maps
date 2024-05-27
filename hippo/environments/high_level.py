import numpy as np
from hippo.environments.w_maze import WmazeTasks
from hippo.agents.prefrontal_agent import LocationRewardMerger

tasks = [
    [0, 1, 2, 1],  # LCRC
    [0, 1, 0, 2],  # LCLR
    [2, 1, 2, 0],  # RCRL
    [0, 1, 0, 1],
    [0, 2, 0, 1],
    [0, 2, 0, 2],
    [0, 2, 1, 2],
    [1, 0, 1, 0],
    [1, 0, 1, 2],
    [1, 0, 2, 0],
    [1, 2, 0, 2],
    [1, 2, 1, 0],
    [1, 2, 1, 2],
    [2, 0, 1, 0],
    [2, 0, 2, 0],
    [2, 0, 2, 1],
    [2, 1, 0, 1],
    [2, 1, 2, 1],
]


class HighLevelWMazeEnvironment:
    def __init__(self, cycle_rule_every, added_task_size=None, n_rules=3):
        self.cycle_rule_every = cycle_rule_every
        self.goal_index = 0
        self.step_count = 0
        self.added_task_size = added_task_size
        self.n_rules = n_rules
        self.set_task(WmazeTasks.TASK0)

    def set_task(self, task_mode):
        self.goal_index = 0
        self.task_mode = task_mode
        self.task_scheme = tasks[task_mode]

        # Make the task longer by adding the mirrored sequence
        if self.added_task_size is not None:
            self.task_scheme = (
                self.task_scheme
                + self.task_scheme[::-1][1 : 1 + self.added_task_size]
            )

    def cycle_rule(self, step):
        if step % self.cycle_rule_every == 0:
            current_idx = self.task_mode.value
            next_idx = (current_idx + 1) % self.n_rules
            new_mode = WmazeTasks(next_idx)
            self.set_task(new_mode)

    @property
    def next_location(self):
        return self.task_scheme[self.goal_index]

    def step(self, action):
        self.step_count += 1
        if action == self.task_scheme[self.goal_index]:
            self.goal_index = (self.goal_index + 1) % len(self.task_scheme)
            reward = 1.0
        else:
            reward = 0.0
        self.cycle_rule(self.step_count)
        return reward


def generate_sequence(
    p_optim=0.0,
    sequence_length=2500,
    cycle_rule_every=1000,
    added_task_size=None,
    n_rules=3,
):
    tokenizer = LocationRewardMerger()

    env = HighLevelWMazeEnvironment(
        cycle_rule_every, added_task_size, n_rules
    )  # 500
    actions = np.random.randint(0, 3, size=sequence_length, dtype=np.int64)
    rewards = np.zeros_like(actions, dtype=np.int64)
    locations = np.zeros_like(actions, dtype=np.int64)
    for i in range(actions.shape[0] - 1):
        # With certain probability choose_optimal action
        if np.random.random() < p_optim:
            actions[i] = env.next_location
        locations[i + 1] = actions[i]
        rewards[i + 1] = env.step(actions[i])
    states = tokenizer.obs_to_state(locations, rewards)
    return states, actions
