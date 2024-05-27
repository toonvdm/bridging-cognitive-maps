from gymnasium.envs.registration import register

from hippo.environments.w_maze import WmazeTasks, get_wmaze
from hippo.environments.action_perception import action_perception_loop

__all__ = ["get_wmaze", "action_perception_loop"]


def register_envs():
    register(
        id="MiniGrid-WMaze3-Task0-v0",
        entry_point="hippo.environments.w_maze:WMazeEnv",
        kwargs={"n_corridors": 3, "task_mode": WmazeTasks.TASK0},
    )

    register(
        id="MiniGrid-WMaze5-Task0-v0",
        entry_point="hippo.environments.w_maze:WMazeEnv",
        kwargs={"n_corridors": 5, "task_mode": WmazeTasks.TASK0},
    )

    register(
        id="MiniGrid-WMaze3-Task1-v0",
        entry_point="hippo.environments.w_maze:WMazeEnv",
        kwargs={"n_corridors": 3, "task_mode": WmazeTasks.TASK1},
    )

    register(
        id="MiniGrid-WMaze5-Task1-v0",
        entry_point="hippo.environments.w_maze:WMazeEnv",
        kwargs={"n_corridors": 5, "task_mode": WmazeTasks.TASK1},
    )
