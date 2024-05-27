import numpy as np
import math
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hippo.agents.replay_agent import ReplayAgent
from hippo.environments.w_maze import WMazeEnv
from hippo.environments.action_perception import action_perception_loop
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from bokeh.layouts import column, row
from bokeh.models import Div

import igraph


def format_ax(a):
    a.set_facecolor("whitesmoke")
    a.grid("on", linestyle="dashed")
    a.set_axisbelow(True)


def add_text(x, text):
    x = x * 255
    cv2.putText(
        x, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(255, 0, 0)
    )
    return x.astype(np.uint8)


def plot_B(B, store_path=None):
    if B.ndim == 2:
        B = B.reshape(*B.shape, 1)
    n_cols = min(B.shape[-1], 5)
    n_rows = math.ceil(B.shape[-1] / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if B.shape[-1] == 1:
        ax = np.array([ax])

    for i, a in enumerate(ax.flatten()[: B.shape[-1]]):
        im = a.imshow(B[..., i])
        divider = make_axes_locatable(a)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
        a.set_title(f"B[..., action={i}]")
        a.set_ylabel("From")
        a.set_xlabel("To")

    [a.axis("off") for a in ax.flatten()[B.shape[-1] :]]
    if store_path is None:
        plt.show()
    else:
        plt.savefig(str(store_path), bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_action_histogram(actions, store_path=None):
    plt.figure(figsize=(1, 1))
    plt.grid("on")
    plt.hist(actions)
    plt.xticks(np.arange(4), ["left", "right", "forward", "rest"], rotation=45)
    if store_path is None:
        plt.show()
    else:
        plt.savefig(
            str(store_path),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()


def replay_and_visualize_trajectory(actions, start_pos=None, env=None):
    """
    Method to visualize the trajectory of an agent.
    We replay the agents actions to get the positions
    """
    if env is None:
        env = WMazeEnv(3, start_pos=start_pos, end=False, agent_view_size=3)
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    logs = action_perception_loop(
        env,
        ReplayAgent(actions),
        record_agent_info=False,
        record_frames=False,
        stop_when_done=False,
    )
    visualize_trajectory(logs["poses"])


def plot_wmaze(ax):
    c = 0.85
    bg = np.zeros((6, 9))
    bg[4, 1:-1] = c
    bg[1:4, 1] = c
    bg[1:4, 4] = c
    bg[1:4, 7] = c

    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, alpha=0.75)

    ax.set_xlim([-0.5, 8.5])
    ax.set_ylim([5, 0])

    ax.set_xticks(np.arange(0.5, 8.0, 1), minor=False)
    ax.set_yticks(np.arange(0.5, 5.0, 1), minor=False)


def visualize_trajectories(
    poses_list, end_states, ax=None, title="", cmap=plt.get_cmap("Dark2")
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_title(title)
    plot_wmaze(ax)
    format_ax(ax)

    alpha = 0.45

    color_index = {k: i for i, k in enumerate(np.unique(end_states))}

    for es in np.unique(end_states):
        ax.plot(
            [-10],
            [-10],
            c=cmap(color_index[es]),
            label=f"State = {es}",
            alpha=alpha,
        )

    # count the amount of end_states
    # j_vals = {t[0][-1][0]: 0 for t in trajectories}
    for i, (poses, end_state) in enumerate(zip(poses_list, end_states)):
        c = cmap(color_index[end_state])

        # Add offset so not all the trajectories are on the same location
        offset = (i / len(poses_list) * 0.80) - 0.40

        px = poses[:, 0] + offset
        py = poses[:, 1] + offset

        # But merge the final point
        # px[-1] -= offset
        py[-1] -= offset

        # ax.plot(px, py, marker=".", c=c, alpha=alpha)
        for pxi, pyi, pxn, pyn in zip(px[:-1], py[:-1], px[1:], py[1:]):
            if abs(pxn - pxi) > 0 or abs(pyn - pyi) > 0:
                ax.arrow(
                    pxi,
                    pyi,
                    pxn - pxi,
                    pyn - pyi,
                    head_width=0.15,
                    color=c,
                    alpha=alpha,
                    length_includes_head=True,
                )

    # add legend
    ax.legend(loc="lower center", ncols=3, fontsize=8)


def visualize_trajectory(poses, title="", colors=None):
    bg = np.zeros((6, 9))
    bg[4, 1:-1] = 1.0
    bg[1:4, 1] = 1.0
    bg[1:4, 4] = 1.0
    bg[1:4, 7] = 1.0
    plt.imshow(bg, cmap="gray", alpha=0.35)
    plt.title(title)

    cmap = plt.get_cmap("Set2")
    format_ax(plt.gca())

    alpha = 1.0
    linecolor = colors
    if colors is None:
        colors = cmap(poses[:, 2])
        plt.plot([-1], c=cmap(0), marker="o", linestyle="None", label="right")
        plt.plot([-1], c=cmap(1), marker="o", linestyle="None", label="down")
        plt.plot([-1], c=cmap(2), marker="o", linestyle="None", label="left")
        plt.plot([-1], c=cmap(3), marker="o", linestyle="None", label="up")
        alpha = 0.33
        linecolor = "black"

    plt.plot(poses[:, 0], poses[:, 1], alpha=alpha, c=linecolor)
    plt.scatter(poses[:, 0], poses[:, 1], c=colors, marker="o", alpha=alpha)
    plt.xlim([0, 8.5])
    plt.ylim([5, 0])

    offsets = {
        0: [0.25, -0.05],  # right
        1: [0.05 * 5 / 8, 0.25 * 5 / 8],  # down
        2: [-0.25, -0.05],  # left
        3: [0.05 * 5 / 8, -0.25 * 5 / 8],  # up
    }

    for i, p in enumerate(poses):
        offset = offsets[p[2]]
        plt.text(p[0] + offset[0], p[1] + offset[1], i, c=cmap(p[2]))

    # add legend
    plt.legend(loc="lower center", ncols=4, fontsize=8)
    plt.tight_layout()


def plot_graph(state_sequence, file_path, agent, to_bokeh=True):
    states = list(np.unique(state_sequence))

    adj = np.zeros((len(states), len(states)), dtype=np.float16)
    for s0, s1 in zip(state_sequence[:-1], state_sequence[1:]):
        adj[states.index(s0), states.index(s1)] = 1.0

    graph = igraph.Graph.Adjacency(adj.tolist())

    cmap = plt.get_cmap("Spectral")
    colors = [cmap(nl)[:3] for nl in np.array(states) / np.max(states)]
    igraph.plot(
        graph,
        file_path,
        bbox=(0, 0, 250, 250),
        layout=graph.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=states,
        vertex_size=25,
        margin=50,
    )
    graph_fig = None
    if to_bokeh:
        h, w = (250, 250)  # make it as heigh as the gif
        graph_fig = column(
            Div(text="<h3>States Graph</h3>"),
            Div(
                text=f"<img src={str(file_path)} width={w} height={h}></img>",
                width=w,
                height=h,
            ),
        )

        legend = ""
        for state in states:
            obs = agent.model_pf.observation_likelihood(state)
            legend += f"state {state} is obs {obs} </br>"
        legend = Div(text=legend)

    return row(graph_fig, legend)
