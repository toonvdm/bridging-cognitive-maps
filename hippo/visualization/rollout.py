from bokeh.models import Div
from bokeh.layouts import row, column
from bokeh.plotting import figure, show, output_file, save

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import pandas as pd


def offset_pos(dataframe, n_policies=None):
    if n_policies is None:
        n_policies = dataframe.policy_idx.max() + 1

    n_sqrt = np.linspace(-0.4, 0.4, int(np.ceil(np.sqrt(n_policies))))
    off_x, off_y = np.meshgrid(n_sqrt, n_sqrt)

    for i in range(off_y.shape[0]):
        off = 2 * ((i / (off_y.shape[0])) - 0.5)
        off_y[:, i] += 0.05 * off + 0.025
        off_x[i, :] += 0.05 * off + 0.025

    repeat = len(dataframe.policy_idx) // n_policies
    off_x = off_x.flatten().repeat(repeat)[: len(dataframe.policy_idx)]
    off_y = off_y.flatten().repeat(repeat)[: len(dataframe.policy_idx)]
    dataframe["offset_pose_x"] = dataframe["pose_x"].to_numpy() + off_x
    dataframe["offset_pose_y"] = dataframe["pose_y"].to_numpy() + off_y
    return dataframe


class VisualizeRollout:
    def __init__(
        self, dataframe, store_path, extra_info=None, title="Rollout"
    ):
        self.data = dataframe
        self.cmap = plt.get_cmap("viridis")
        output_file(store_path, title=title)
        self.title = title
        self.store_path = store_path.parent

        # print(self.data.head())

        self.efe_keys = ["infogain", "expected_utility"]

        self.extra_info = extra_info
        if self.extra_info is None:
            self.extra_info = dict()

    def add_wmaze(self, p):
        px = [0.5, 0.5, 7.5, 7.5, 6.5, 6.5, 4.5, 4.5, 3.5, 3.5, 1.5, 1.5, 0.5]
        py = [0.5, 4.5, 4.5, 0.5, 0.5, 3.5, 3.5, 0.5, 0.5, 3.5, 3.5, 0.5, 0.5]
        p.line(px, py, color="black")
        b, left = np.meshgrid(np.arange(0.5, 4.5, 1), np.arange(0.5, 7.5, 1.0))
        b = b.flatten()[::2]
        left = left.flatten()[::2]
        t = 1 + b
        r = 1 + left
        for i in range(7):
            off = int(i % 2 == 1)
            b[i * 2 : (i + 1) * 2] = b[i * 2 : (i + 1) * 2] + off
            t[i * 2 : (i + 1) * 2] = t[i * 2 : (i + 1) * 2] + off
            if i in [1, 2, 4, 5]:
                b[i * 2 : (i + 1) * 2 - off] = 0
                t[i * 2 : (i + 1) * 2 - off] = 0
                left[i * 2 : (i + 1) * 2 - off] = 0
                r[i * 2 : (i + 1) * 2 - off] = 0
        p.quad(bottom=b, top=t, left=left, right=r, alpha=0.125)
        return p

    def figure_trajectory(self, pose):
        p = figure(title="Trajectory", match_aspect=True, height=500)

        x_off = np.zeros(pose[:, 0].shape)
        y_off = np.zeros(pose[:, 1].shape)
        general_off = (
            0.30 * np.arange(pose[:, 1].shape[0]) / pose[:, 1].shape[0]
        ) - 0.15
        for i, direction in enumerate(pose[:, 2]):
            if direction == 0:
                x_off[i] -= 0.25
            elif direction == 1:
                y_off[i] += 0.25
            elif direction == 2:
                x_off[i] -= 0.25
            else:
                y_off[i] -= 0.25

        px = pose[:, 0] + x_off + general_off
        py = pose[:, 1] + y_off + general_off
        p.line(px, py)
        p.y_range.flipped = True

        p.scatter(px[:1], py[:], legend_label="start")
        p.scatter(px[-1:], py[-1:], color="firebrick", legend_label="end")

        goal = self.extra_info.get("goal_pos", None)
        if goal is not None:
            p.circle(
                x=goal[0] - 0.30,
                y=goal[1] - 0.30,
                size=10,
                fill_color="green",
                legend_label="goal",
            )
        self.add_wmaze(p)
        return p

    def figure_maze_with_value(self, k, policies_of_interest=None):
        p = figure(title=k, match_aspect=True, height=500)
        if policies_of_interest is None:
            policies_of_interest = self.data["policy_idx"].unique()

        filtered_pd = []
        for i in policies_of_interest:
            filtered_pd.append(self.data[self.data["policy_idx"] == i])
        filtered_pd = pd.concat(filtered_pd)
        filtered_pd = offset_pos(filtered_pd, len(policies_of_interest))

        for i in policies_of_interest:
            filtered = filtered_pd[filtered_pd["policy_idx"] == i]
            p.line(filtered["offset_pose_x"], filtered["offset_pose_y"])

            p.scatter(
                filtered["offset_pose_x"],
                filtered["offset_pose_y"],
                color=self.cmap(filtered[k].to_numpy()),
            )
            p.y_range.flipped = True

        self.add_wmaze(p)
        return p

    def figure_value(self, policy_index, k):
        off = 1.15 * (self.data[k].max() - self.data[k].min())
        y_range = (self.data[k].min() - off, self.data[k].max() + off)
        filtered = self.data[self.data["policy_idx"] == policy_index]
        value = filtered[k].to_numpy()
        p = figure(
            title=k,
            y_range=y_range,
            height=200,
            toolbar_location=None,
            tools="",
        )
        p.scatter(np.arange(len(value)), value)
        p.line(np.arange(len(value)), value)
        return p

    def figure_bar(self, policy_index):
        n = (
            self.data[self.data["policy_idx"] == policy_index]["action"]
            .to_numpy()
            .shape[0]
        )
        keys = ["negative_expected_free_energy"] + self.efe_keys
        mini = np.min([self.data[k].min() for k in keys])
        maxi = np.max([self.data[k].max() for k in keys])
        p = figure(
            title="Expected Free Energy",
            height=200,
            x_range=keys,
            toolbar_location=None,
            tools="",
            y_range=(n * mini, n * maxi),
        )
        filtered = self.data[self.data["policy_idx"] == policy_index]
        top = [filtered[k].sum() for k in keys]
        p.vbar(x=keys, top=top, width=0.9)

        return p

    def figure_actions(self, policy_index):
        filtered = self.data[self.data["policy_idx"] == policy_index]
        label = ["turn left", "turn right", "forward", "rest"]
        actions = filtered["action"].to_numpy()
        p = figure(
            title=f"actions policy {policy_index}",
            height=200,
            toolbar_location=None,
            y_range=(0.5, 2.0),
            tools="",
        )
        for i, action in enumerate(actions):
            if action == 0:
                p.triangle(
                    x=i,
                    y=1,
                    size=25,
                    fill_color="purple",
                    legend_label=label[action],
                )
            elif action == 1:
                p.inverted_triangle(
                    x=i,
                    y=1,
                    size=25,
                    fill_color="blue",
                    legend_label=label[action],
                )
            elif action == 2:
                p.dash(
                    x=i,
                    y=1,
                    size=25,
                    fill_color="lightgreen",
                    legend_label=label[action],
                )
            elif action == 3:
                p.circle_x(
                    x=i,
                    y=1,
                    size=25,
                    fill_color="firebrick",
                    legend_label=label[action],
                )

        return p

    def row_figure_policy(self, policy_index):
        action_fig = self.figure_actions(policy_index)
        figs = []
        for k in self.efe_keys:  #  + ["state_entropy"]:
            figs.append(self.figure_value(policy_index, k))
        bar_fig = self.figure_bar(policy_index)

        row_1 = row(action_fig, bar_fig)
        row_2 = row(figs)
        return column(
            Div(text=f"<h2>Policy {policy_index}</h2>"), row_1, row_2
        )

    def text_info(self):
        keys = ["negative_expected_free_energy"] + self.efe_keys

        style = "<style> th {padding-right: 15px;} td {padding-right:15px;}</style>"

        table = ["<table>"]
        row = "<tr><th>Key</th><th>Lowest Policy</th><th>Highest Policy</th>"
        table.append(row)
        table.append("<br>")
        policies_of_interest = []
        for k in keys:
            value = self.data.groupby("policy_idx").sum(k)[k].to_numpy()
            row = f"<tr><td>{k}</td><td>{value.argmin()}</td><td>{value.argmax()}</td>"
            table.append(row)

            policies_of_interest.append(value.argmin())
            policies_of_interest.append(value.argmax())

        table.append("</table>")

        return Div(text=style + "".join(table)), sorted(
            list(set(policies_of_interest))
        )

    def img_to_bokeh(self, img):
        img = Image.fromarray(img).convert("RGBA")
        xdim, ydim = img.size
        img2 = np.empty((ydim, xdim), dtype=np.uint32)
        view = img2.view(dtype=np.uint8).reshape((ydim, xdim, 4))
        view[:, :, :] = np.flipud(np.asarray(img))

        # Display the 32-bit RGBA image
        fig = figure(
            title="Observation",
            x_range=(0, xdim),
            y_range=(0, ydim),
            height=6 * ydim,
            width=6 * xdim,
            toolbar_location=None,
            tools="",
        )
        fig.image_rgba(image=[img2], x=0, y=0, dw=xdim, dh=ydim)
        return fig

    def row_extra_info(self):
        items = []

        if self.extra_info.get("selected", None) is not None:
            i = Div(
                text=f"Selected policy index: {self.extra_info['selected']}"
            )
            items.append(i)

        optimal = self.extra_info.get("optimal", None)
        if optimal is not None:
            i = Div(
                text=f"Optimal (direct path to goal) policy index: {optimal}"
            )
            items.append(i)

        obs = self.extra_info.get("observation", None)
        if obs is not None:
            i = self.img_to_bokeh(obs.astype(np.uint8))
            items.append(i)

        qs = self.extra_info.get("qs_clones", None)
        if qs is not None:
            keys = [str(i) for i in np.arange(len(qs))]
            p = figure(
                title="Q(s) Only belief over clone states",
                height=200,
                toolbar_location=None,
                tools="",
                x_range=keys,
            )
            p.vbar(x=keys, top=qs)
            items.append(p)

        q_pi = self.extra_info.get("q_pi", None)
        if qs is not None:
            keys = [str(i) for i in np.arange(len(q_pi))]
            p = figure(
                title="Q(pi)",
                height=200,
                toolbar_location=None,
                tools="",
                x_range=keys,
            )
            p.vbar(x=keys, top=q_pi)
            items.append(p)

        c = self.extra_info.get("constraint", None)
        if c is not None:
            keys = [str(i) for i in np.arange(len(c))]
            p = figure(
                title="Constraint",
                x_range=keys,
                width=3000,
                height=300,
            )
            p.vbar(x=keys, top=c)
            items.append(p)

        states = self.extra_info.get("states", None)
        if states is not None:
            p = Div(text=f"States: {' - '.join([str(s) for s in states])}")
            items.append(p)

        pose = self.extra_info.get("pose", None)
        if pose is not None:
            figure_trajectory = self.figure_trajectory(pose)
            items.append(figure_trajectory)

        return column(*items)

    def plot(self, show_page=True):
        extra_info = self.row_extra_info()

        text_info, poi = self.text_info()

        poi.append(self.extra_info.get("selected", poi[0]))
        poi.append(self.extra_info.get("optimal", poi[0]))
        poi = sorted(list(set(poi)))

        # figs = []
        # for k in self.efe_keys:
        #     figs.append(self.figure_maze_with_value(k, poi))
        # row_1 = row(*figs)

        policy_figs = []
        for i in poi:
            policy_figs.append(self.row_figure_policy(i))

        fig = column(
            Div(text=f"<h1>Rollout of episode: {self.title}</h1>"),
            extra_info,
            text_info,
            # row_1,
            *policy_figs,
        )
        if show_page:
            show(fig)
        else:
            save(fig)
