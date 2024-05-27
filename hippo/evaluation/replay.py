import numpy as np
import matplotlib.pyplot as plt

from hippo.environments.w_maze import get_wmaze
from hippo.environments.action_perception import action_perception_loop
from hippo.environments.sequences import EvaluationSequences3x3WMaze
from hippo.visualization import visualize_trajectories, plot_B
from hippo.agents.replay_agent import ReplayAgent

import yaml

from copy import copy

import logging

logger = logging.getLogger(__name__)


class ReplayEvaluation:
    """
    Class for evaluating CSCG agents by evaluating state inference after
    replaying known trajectories to different goals
    """

    def __init__(self, store_path, debug_sequences=True):
        self.sequences = EvaluationSequences3x3WMaze(
            debug_mode=debug_sequences
        )
        self.store_path = store_path

    def _replay_sequence(self, goal_index, start_index):
        env = get_wmaze(self.sequences.start_positions[start_index])

        actions = self.sequences.action_sequences[goal_index][start_index]
        run_logs = action_perception_loop(
            env,
            ReplayAgent(copy(actions) + [2]),
            len(actions) + 1,
            record_frames=False,
            record_agent_info=False,
            progress_bar=False,
        )
        return run_logs

    def _decode_belief(self, agent_ui, run_logs):
        agent_ui.reset()
        for o, a in zip(run_logs["obs"], run_logs["action"]):
            qs = agent_ui.replay_step(o, a)
        # Add zeroth dimension because of pymdp legacy: TODO refactor it all
        return [np.array([qs])]

    def _visualize_trajectories(self, trajectories, run_index):
        """
        Visualize the replayed trajectories in the color of the end state, i.e. show
        how many clones are needed to encode each of the individual corridors
        """
        n_layers = len(
            list([k for k in trajectories[0].keys() if "end_state" in k])
        )

        fig, ax = plt.subplots(1, n_layers, figsize=(n_layers * 8, 16))
        if n_layers == 1:
            ax = np.array([ax])

        for layer_index, a in enumerate(ax.flatten()):
            poses_list = [t["pose"] for t in trajectories]
            end_states = [t[f"end_state_{layer_index}"] for t in trajectories]
            visualize_trajectories(
                poses_list,
                end_states,
                ax=a,
                title=f"Trajectories colored by layer {layer_index} end-state",
            )
        plt.tight_layout()
        plt.savefig(
            self.store_path
            / f"inference_trajectories_run_{run_index:02d}.png",
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    def _n_states(self, trajectories, run_index):
        """
        Compute the amount of states used to encode each corridor, and in total
        """
        n_layers = len(
            list([k for k in trajectories[0].keys() if "end_state" in k])
        )

        n_states = {}
        for layer_index in range(n_layers):
            key = f"layer_{layer_index}"
            n_states[key] = {}
            all_states = []
            for goal_index in self.sequences.goal_positions.keys():
                states = np.unique(
                    [t[f"end_state_{layer_index}"] for t in trajectories]
                )
                n_states[key][goal_index] = len(states)
                all_states += list(states)

            n_states[key]["all"] = len(np.unique(all_states))

        filename = str(self.store_path / f"n_states_run_{run_index:02d}.yml")
        with open(filename, "w") as outfile:
            yaml.dump(n_states, outfile, default_flow_style=False)

        return n_states

    def _visualize_b(self, agent_ui, run_index):
        plot_B(
            agent_ui.T.transpose(2, 1, 0),
            self.store_path / f"transition_matrix_episode_{run_index:02d}.png",
        )

    def evaluate_agent(self, agent_ui, run_index=0):
        """
        :param agent_ui: CSCGAgent or HierarchicAlagentV2 under investigation,
            crucially the agent should entail an infer_states method that uses
            agent.obs and agent.acs for state inference.
            # TODO: abstract this away in the class
        :run int: run index, used for creating the store_files
        """
        logger.info(f"Running inference evaluate for run {run_index:02d}")

        trajectories = []
        for goal_index in self.sequences.goal_positions.keys():
            for start_index in self.sequences.start_positions.keys():
                # Replay the actions of the sequence
                run_logs = self._replay_sequence(goal_index, start_index)
                # Use the logs to decode into a belief over the end state
                beliefs = self._decode_belief(agent_ui, run_logs)

                # Store the intermediate results in a trajectories dictionary
                trajectories.append(
                    {
                        "pose": run_logs["pose"],
                        "goal_index": goal_index,
                        "start_index": start_index,
                    }
                )
                trajectories[-1].update(
                    {
                        f"end_state_{i}": b[0].argmax()
                        for i, b in enumerate(beliefs)
                    }
                )

        self._visualize_trajectories(trajectories, run_index)
        n_states = self._n_states(trajectories, run_index)
        self._visualize_b(agent_ui, run_index=run_index)
        return n_states
