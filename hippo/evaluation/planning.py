import numpy as np
import matplotlib.pyplot as plt

from hippo.environments.w_maze import get_wmaze
from hippo.environments.action_perception import action_perception_loop
from hippo.visualization import visualize_trajectories

from hippo.evaluation.replay import ReplayEvaluation

import logging

from copy import copy
from hippo.agents.replay_agent import ReplayAgent
import pandas as pd
from hippo.visualization.rollout import VisualizeRollout

from bokeh.layouts import column
from bokeh.plotting import output_file, save
from bokeh.models import Div

logger = logging.getLogger(__name__)


def rollout_policy(policy_df, start):
    actions = copy(policy_df["action"].to_list())

    logs = action_perception_loop(
        get_wmaze(start),
        ReplayAgent(actions),
        len(actions),
        record_frames=False,
        record_agent_info=False,
        progress_bar=False,
    )

    poses = logs["pose"]
    policy_df["pose_x"] = poses[:, 0]
    policy_df["pose_y"] = poses[:, 1]
    policy_df["pose_dir"] = poses[:, 2]

    return policy_df


def policy_to_df(policies, policy_idx, keys=None):
    if keys is None:
        keys = ["expected_utility", "infogain", "action", "state_entropy"]

    d = dict({"policy_idx": policy_idx})
    for k in keys:
        d.update({k: [p.data[k] for p in policies[policy_idx]]})

    d["negative_expected_free_energy"] = np.array(
        d["expected_utility"]
    ) + np.array(d["infogain"])
    return pd.DataFrame.from_dict(d)


def policies_to_df(policies, start):
    dataframes = []
    for i in range(len(policies)):
        data = policy_to_df(policies, i)
        data = rollout_policy(data, start)
        dataframes.append(data)
    return pd.concat(dataframes)


class PlannerEvaluation(ReplayEvaluation):
    """
    Class for evaluating CSCG agents by investigating planning behavior when
    reaching certain goal states.
    """

    def success(self, goal, poses):
        return np.any([np.all(p == goal) for p in poses])

    def _evaluate_run(self, run_logs, goal, start):
        n_steps = len(run_logs["pose"])
        success = self.success(goal, run_logs["pose"])
        if success:
            reward_proxy = [np.all(p == goal) for p in run_logs["pose"]]
            n_steps = np.nonzero(reward_proxy)[0][0] + 1

        eval_logs = {
            "pose": run_logs["pose"],
            "success": success,
            "n_steps": n_steps,
            "start": start,
            "goal": goal,
        }

        return eval_logs

    def _visualize_instrumental_episodes(self, episode_logs, run_index):
        n_goals = len(list(episode_logs.keys()))
        fig, ax = plt.subplots(1, n_goals, figsize=(n_goals * 8, 16))
        for a, goal_index in zip(ax.flatten(), episode_logs.keys()):
            poses_list = [
                epi["pose"][: epi["n_steps"]]
                for epi in episode_logs[goal_index]
            ]
            success_val = [
                1.0 * epi["success"] for epi in episode_logs[goal_index]
            ]

            visualize_trajectories(
                poses_list,
                success_val,
                ax=a,
                title="Trajectories colored by success",
                cmap=plt.get_cmap("RdYlGn"),
            )
        plt.tight_layout()
        plt.savefig(
            self.store_path
            / f"instrumental_trajectories_run_{run_index:02d}.png",
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

    def investigate_rollout(
        self, agent_ui, run_index, n_steps=1, goal_index=0, start_index=0
    ):
        # take one run to investigate + start it from a location where we know
        # the state inference is correct (i.e. unambiguous T-junction)
        # 64 (4, 4, 3)
        index = self.sequences.start_poses_to_index[(4, 4, 3)]
        run_logs = self._replay_sequence(goal_index, index)
        beliefs = self._decode_belief(agent_ui, run_logs)

        agent_ui.set_state_preference(beliefs[-1][0].argmax())

        start = self.sequences.start_positions[start_index]
        goal = self.sequences.goal_positions[goal_index]

        logs = action_perception_loop(
            get_wmaze(start),
            agent_ui,
            n_steps,
            record_frames=False,
            record_agent_info=True,
        )

        store_path = (
            self.store_path / f"{goal_index}_from_{start_index}_{run_index}"
        )
        store_path.mkdir(exist_ok=True, parents=True)
        for idx in [n_steps - 1]:  # range(n_steps):
            tree = logs["agent_info"][idx]["efe"]["tree"]

            replay_from = logs["pose"][idx]

            dataframe = policies_to_df(tree, replay_from)
            states = [l["state"] for l in logs["agent_info"]]
            extra_info = {
                "observation": np.hstack(logs["obs"][: idx + 1]),
                "selected": logs["agent_info"][idx]["efe"]["selected"],
                "qs_clones": logs["agent_info"][idx]["qs_clones"],
                "pose": logs["pose"][: idx + 1],
                "goal_pos": self.sequences.goal_positions[goal_index],
                "states": states[: idx + 1],
                "constraint": logs["agent_info"][idx]["efe"]["constraint"],
                "q_pi": logs["agent_info"][idx]["efe"]["q_pi"],
            }

            vis = VisualizeRollout(
                dataframe,
                store_path / f"rollout_{idx}.html",
                extra_info,
                title=f"{start_index} to {goal_index} ({idx})",
            )
            vis.plot(show_page=False)

        success = self.success(goal, logs["pose"])
        return vis.figure_trajectory(extra_info["pose"]), success

    def publish_trajectories(self, trajects):
        output_file(
            self.store_path / "trajectories.html", title="Trajectories"
        )
        save(column(*trajects))

    def evaluate_agent(self, agent_ui, run_index=0):
        success_trajects = []
        failure_trajects = []
        goal_positions = self.sequences.goal_positions
        start_positions = self.sequences.start_positions
        for start_index, start_node in start_positions.items():
            for goal_index, goal_node in goal_positions.items():
                traject, success = self.investigate_rollout(
                    agent_ui,
                    run_index=run_index,
                    n_steps=30,
                    start_index=start_index,
                    goal_index=goal_index,
                )

                col = column(
                    Div(
                        text=f"<b>{start_index}</b>: {start_node} to <b>{goal_index}</b>: {goal_node}"
                    ),
                    traject,
                )
                if success:
                    success_trajects.append(col)
                else:
                    failure_trajects.append(col)

                ns, nf = len(success_trajects), len(failure_trajects)

                s_to_g = (
                    f"{start_index} {start_node} -> {goal_index} {goal_node}"
                )
                logger.info(f"{s_to_g}: {'success' if success else 'failure'}")
                logger.info(f"{ns}/{ns+nf} Successes")

        ns, nf = len(success_trajects), len(failure_trajects)
        failure_trajects = [
            Div(text=f"<h2>Failure Trajectories ({nf}/{ns+nf})<h2>")
        ] + failure_trajects
        success_trajects = [
            Div(text=f"<h2>Success Trajectories ({ns}/{ns+nf})<h2>")
        ] + success_trajects
        title = Div(
            text="<h1>All trajectories</h1><p>0: right</br>1: down</br>2: left</br>3: up</p>"
        )
        self.publish_trajectories(
            [title] + failure_trajects + success_trajects
        )
