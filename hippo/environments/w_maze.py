import numpy as np

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from enum import IntEnum


class WmazeTasks(IntEnum):
    TASK0 = 0
    TASK1 = 1
    TASK2 = 2
    TASK3 = 3
    TASK4 = 4
    TASK5 = 5
    TASK6 = 6
    TASK7 = 7
    TASK8 = 8
    TASK9 = 9
    TASK10 = 10
    TASK11 = 11
    TASK12 = 12
    TASK13 = 13
    TASK14 = 14
    TASK15 = 15
    TASK16 = 16
    TASK17 = 17
    TASK18 = 18


def get_wmaze(
    start=None,
    end=False,
    agent_view_size=3,
    n_corridors=3,
    task_mode=WmazeTasks.TASK0,
    cycle_rules_every=None,
    cycle_rules_random=False,
    n_rules=1,
):
    env = WMazeEnv(
        n_corridors=n_corridors,
        start_pose=start,
        end=end,
        agent_view_size=agent_view_size,
        task_mode=task_mode,
        cycle_rules_every=cycle_rules_every,
        cycle_rules_random=cycle_rules_random,
        n_rules=n_rules,
    )
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    return env


class WMazeEnv(MiniGridEnv):
    """
    ## Description

    This environment is a recreation of the W-maze experiment conducted
    on rodents for rule learning

    ## Mission space

    Task 0: LCRC
    Task 1: LCLR
    Task 2: RCRL

    ## Action Space
    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Unused                    |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The environment terminates when all goals are reached in correct order,
    if the agent steps on lava, or if the agent has taken more steps than
    the max steps parameter.

    ## Configurations

    MiniGrid-WMaze3-Task0-v0
    MiniGrid-WMaze5-Task0-v0
    MiniGrid-WMaze3-Task1-v0
    MiniGrid-WMaze5-Task1-v0
    """

    def __init__(
        self,
        n_corridors=3,
        see_through_walls=False,
        max_steps=None,
        task_mode=WmazeTasks.TASK1,
        visual_markers=False,
        failure_possible=False,
        start_pose=None,
        end=False,
        stop_at_first=False,
        cycle_rules_every=None,
        cycle_rules_random=False,
        added_task_size=None,
        n_rules=3,
        **kwargs,
    ):
        self.n_cols = n_corridors
        size = n_corridors * 3

        self.size = size
        if max_steps is None:
            max_steps = 2 * 5 * size**2

        mission_space = MissionSpace(mission_func=self._gen_mission)
        MiniGridEnv.__init__(
            self,
            mission_space=mission_space,
            width=size,
            height=6,
            see_through_walls=see_through_walls,
            max_steps=max_steps,
            **kwargs,
        )

        self.visual_markers = visual_markers
        self.task_mode = task_mode
        self.failure_possible = failure_possible
        self.start_pose = start_pose

        self.end = end
        self.stop_at_first = stop_at_first

        self.n_rules = n_rules
        self.task_scheme = None
        self.added_task_size = added_task_size
        self.set_task(task_mode)

        self.prev_index = 0
        self.goal_index = 1

        self.cycle_rules_every = cycle_rules_every
        self.cycle_rules_random = cycle_rules_random

        # If random start
        self.possible_starts = []
        for k in range(4):
            for j in range(n_corridors):
                for i in range(3):
                    self.possible_starts.append((j * 3 + 1, i + 1, k))
            # bottom row
            for j in range(n_corridors * 3 - 2):
                self.possible_starts.append((j + 1, 4, k))

    def set_task(self, task_mode):
        # All combinations of visiting the three corridors in
        # Four steps, without repeating a visit
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

        self.goal_index = np.random.randint(4)
        self.prev_index = self.goal_index - 1
        self.task_mode = task_mode

        self.task_scheme = tasks[task_mode]

        # Make the task longer by adding the mirrored sequence
        if self.added_task_size is not None:
            self.task_scheme = (
                self.task_scheme
                + self.task_scheme[::-1][1 : 1 + self.added_task_size]
            )

    def cycle_rule(self, step):
        if step % self.cycle_rules_every == 0:
            current_idx = self.task_mode.value  # - 1
            if self.cycle_rules_random and self.n_rules > 1:
                next_idx = current_idx
                while next_idx == current_idx:
                    next_idx = np.random.randint(0, self.n_rules)
            else:
                next_idx = (current_idx + 1) % self.n_rules

            new_mode = WmazeTasks(next_idx)
            self.set_task(new_mode)

    @staticmethod
    def _gen_mission():
        return "Find the reward in the maze"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # x from left, y from top
        # Generate corridors
        self.corridor_locations = []
        for idx in range(self.size // 3):
            self.corridor_locations += [(idx * 3 + 1, 1)]
            self.grid.vert_wall(idx * 3, 1, height - 3)
            self.grid.vert_wall(idx * 3 + 2, 1, height - 3)

        self.grid.vert_wall(0, height - 2, 1)
        self.grid.vert_wall(width - 1, height - 2, 1)

        self.grid.horz_wall(0, 0, width)
        self.grid.horz_wall(0, height - 1, width)

        # # Fix the player's start position and orientation
        if self.start_pose is not None:
            if isinstance(self.start_pose, int):
                pose = (*self.corridor_locations[self.start_pose], 1)
            else:
                pose = self.start_pose
        else:
            idx = np.random.randint(len(self.possible_starts))
            pose = self.possible_starts[idx]

        self.agent_pos = pose[:2]
        self.agent_dir = pose[-1]

        # set the rule
        self.previous_goal = None

        (
            self.success_positions,
            self.failure_positions,
        ) = self.update_success_failure_positions()
        self.put_goals()

    def put_goals(self):
        if self.visual_markers:
            for s in self.corridor_locations:
                self.grid.set(*s, None)
            for s in self.success_positions:
                self.put_obj(Goal(), *s)
            for s in self.failure_positions:
                self.put_obj(Lava(), *s)

    def update_success_failure_positions(self):
        self.prev_index = self.goal_index
        self.goal_index = (self.goal_index + 1) % len(self.task_scheme)

        positions = [
            self.corridor_locations[self.task_scheme[self.goal_index]]
        ]

        # All other positions are failures
        if self.failure_possible:
            failure_positions = [
                c
                for c in self.corridor_locations
                if c not in positions + [self.agent_pos]
            ]
        else:
            failure_positions = []

        return positions, failure_positions

    def step(self, action):
        obs, reward, terminated, truncated, info = MiniGridEnv.step(
            self, action
        )
        # Read the light of the previous location, before setting it anew

        if tuple(self.agent_pos) in self.success_positions:
            reward = self._reward()

            (
                self.success_positions,
                self.failure_positions,
            ) = self.update_success_failure_positions()
            self.put_goals()

            terminated = len(self.success_positions) == 0

            if self.stop_at_first:
                terminated = True

        if tuple(self.agent_pos) in self.failure_positions:
            reward = 0
            terminated = True

        if self.cycle_rules_every is not None:
            self.cycle_rule(self.step_count)

        pose = np.array([*self.agent_pos, self.agent_dir])

        light = self.task_scheme[self.prev_index]
        info.update(
            {
                "pose": pose,
                "rule": self.task_mode.value,
                "inbound": (
                    self.success_positions[0] == self.corridor_locations[1]
                ),
                "prev": light,
            }
        )

        return obs, reward, terminated, truncated, info
