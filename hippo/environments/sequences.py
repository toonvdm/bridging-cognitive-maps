import networkx as nx
import numpy as np
from functools import cached_property
from hippo.environments.w_maze import get_wmaze


def transition(from_pose, action, valid_poses):
    direction_to_movement = {
        0: np.array([1, 0]),  # right
        1: np.array([0, 1]),  # down
        2: np.array([-1, 0]),  # left
        3: np.array([0, -1]),  # up
    }

    new_pose = np.array(from_pose)
    if action == 0:
        # Turn left
        new_pose[2] = (new_pose[2] - 1) % 4

    elif action == 1:
        # Turn right
        new_pose[2] = (new_pose[2] + 1) % 4

    elif action == 2:
        # go forward
        new_pose[:2] += direction_to_movement[from_pose[2]]

    new_pose = tuple(new_pose)

    if new_pose not in valid_poses:
        new_pose = from_pose

    new_pose_index = valid_poses.index(new_pose)
    return new_pose, new_pose_index


def create_transition_matrix(valid_poses):
    transition_matrix = np.zeros((3, len(valid_poses), len(valid_poses)))
    for action in range(transition_matrix.shape[0]):
        for from_index, from_pose in enumerate(valid_poses):
            _, to_index = transition(from_pose, action, valid_poses)
            transition_matrix[action, from_index, to_index] = 1.0
    return transition_matrix


def get_path(graph, from_node, to_node, valid_poses):
    from_index = valid_poses.index(from_node)
    to_index = valid_poses.index(to_node)
    path = nx.shortest_path(graph, source=from_index, target=to_index)
    return path


def get_random_path(graph, from_node, to_node, valid_poses):
    from_index = valid_poses.index(from_node)
    to_index = valid_poses.index(to_node)
    paths = nx.all_simple_paths(graph, source=from_index, target=to_index, cutoff=15)
    paths = list(paths)
    if len(paths) > 0:
        selected = np.random.choice(np.arange(len(paths)))
        path = paths[selected]
    else:
        path = [from_node]
    return path


def path_to_actions(path, t_mat):
    actions = np.zeros(len(path) - 1, dtype=np.int8)
    for i, (from_idx, to_idx) in enumerate(zip(path[:-1], path[1:])):
        actions[i] = t_mat[:, from_idx, to_idx].argmax()
    return actions


class EvaluationSequences3x3WMaze:
    """
    Some ground truth trajectories that reach the end of the three corridors
    for the 3x3 W-maze
    """

    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

        self.t_mat = create_transition_matrix(self.valid_poses)
        self.a_mat = self.t_mat.sum(axis=0) > 0
        self.graph = nx.from_numpy_array(self.a_mat, create_using=nx.DiGraph)

    @cached_property
    def valid_poses(self):
        return get_wmaze().possible_starts

    @cached_property
    def start_positions(self):
        valid_poses = {i: pose for i, pose in enumerate(self.valid_poses)}

        if self.debug_mode:
            # Only return bottom row poses to have a smaller evaluation set
            # filtered = [f for f, p in valid_poses.items() if p[1] == 4]
            # valid_poses = {f: valid_poses[f] for f in filtered}

            # Only return hallway poses to have difficult set
            filtered = [f for f, p in valid_poses.items() if p[1] != 4]
            # T-junction
            filtered.append([f for f, p in valid_poses.items() if p == (4, 4, 3)][0])
            valid_poses = {f: valid_poses[f] for f in filtered}

        return valid_poses

    @cached_property
    def start_poses_to_index(self):
        return {v: k for k, v in self.start_positions.items()}

    @cached_property
    def goal_positions(self):
        return {i: (1 + i * 3, 1, 3) for i in range(3)}

    @cached_property
    def action_sequences(self):
        sequences = dict()
        for goal_index, goal_node in self.goal_positions.items():
            if sequences.get(goal_index, None) is None:
                sequences[goal_index] = dict()
            for start_index, start_node in self.start_positions.items():
                path = get_path(self.graph, start_node, goal_node, self.valid_poses)
                actions = path_to_actions(path, self.t_mat)
                sequences[goal_index][start_index] = list(actions)
        return sequences
