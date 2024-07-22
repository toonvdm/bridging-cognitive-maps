import networkx as nx
import numpy as np
from copy import copy
import random
import logging

from hippo.environments.w_maze import get_wmaze
from hippo.agents.replay_agent import ReplayAgent
from hippo.environments.action_perception import action_perception_loop
from hippo.environments.sequences import (
    create_transition_matrix,
    EvaluationSequences3x3WMaze,
)
from hippo.environments.sequences import (
    get_path,
    path_to_actions,
    get_random_path,
)
from hippo import get_store_path, get_recents_path, save_config

import imageio as io

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    seed = 42

    logging.basicConfig(level=logging.INFO)
    store_path = get_store_path("full-exploration-data")

    dataset_path = store_path / "full_exploration.npz"

    logger.info(f"Storing results at: {store_path}")

    save_config(
        {"dataset_path": str(dataset_path)},
        get_recents_path() / "dataset_path.yml",
    )

    valid_poses = get_wmaze().possible_starts

    t_mat = create_transition_matrix(valid_poses)
    a_mat = t_mat.sum(axis=0) > 0
    graph = nx.from_numpy_array(a_mat, create_using=nx.DiGraph)

    # reproducability
    np.random.seed(seed)
    random.seed(seed)

    sequences = EvaluationSequences3x3WMaze()

    # Start in the center!
    start_index = sequences.start_poses_to_index[(4, 1, 3)]
    pose_index_list = [(i, p) for i, p in enumerate(valid_poses)]

    path = [start_index]
    for start_index, start in copy(pose_index_list):
        path += get_path(graph, valid_poses[path[-1]], start, valid_poses)[1:]
        np.random.shuffle(pose_index_list)
        for goal_index, goal in pose_index_list:
            path += get_path(graph, valid_poses[path[-1]], goal, valid_poses)[
                1:
            ]
            path += get_random_path(graph, goal, start, valid_poses)[1:]

    actions = path_to_actions(path, t_mat).tolist()

    agent = ReplayAgent(actions=actions)
    data = action_perception_loop(
        get_wmaze(start=valid_poses[path[0]]),
        agent,
        len(agent.actions),
        record_frames=False,
        record_agent_info=False,
        progress_bar=True,
    )

    np.savez(
        dataset_path,
        **{k: v for k, v in data.items() if k != "frames"},
    )

    if len(data["frames"]) > 1:
        io.mimwrite(store_path / "frames.gif", data["frames"])
