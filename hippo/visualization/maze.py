import numpy as np
import matplotlib.pyplot as plt


def add_wmaze_plot(ax, color="tab:orange"):
    px = [0.5, 0.5, 7.5, 7.5, 6.5, 6.5, 4.5, 4.5, 3.5, 3.5, 1.5, 1.5, 0.5]
    py = [0.5, 4.5, 4.5, 0.5, 0.5, 3.5, 3.5, 0.5, 0.5, 3.5, 3.5, 0.5, 0.5]

    ax.text(1, 0.3, "L", ha="center", fontsize=10, color="black")
    ax.text(4, 0.3, "C", ha="center", fontsize=10, color="black")
    ax.text(7, 0.3, "R", ha="center", fontsize=10, color="black")

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

    for bi, ti, li, ri in zip(b, t, left, r):
        rect = plt.Rectangle((li, bi), ri - li, ti - bi, fc=color, alpha=0.5)

        ax.add_patch(rect)
    ax.plot(px, py, color="black")
    ax.set_ylim(ax.get_ylim()[::-1])


def add_trial(trial_idx, data, ax):
    filtered = data[data["agent_name"] == "CSCG_LOC_RULE0"]
    filtered = filtered[filtered["control_mode"] == "control"]
    filtered = filtered[filtered["trial"] == trial_idx]

    x, y = filtered["pos_x"].to_numpy().astype(np.float32), filtered[
        "pos_y"
    ].to_numpy().astype(np.float32)
    x += np.random.randn(*x.shape) * 0.075
    y += np.random.randn(*x.shape) * 0.075

    c = plt.get_cmap("Reds")(np.arange(len(x)) / len(x))
    for i, (x1, x2, y1, y2) in enumerate(zip(x[:-1], x[1:], y[:-1], y[1:])):
        ax.plot([x1, x2], [y1, y2], c=c[i], alpha=i / len(x), linestyle="--")


def add_single(trial_idx, data, ax, from_x, to_x):
    filtered = data[data["agent_name"] == "CSCG_LOC_RULE0"]
    filtered = filtered[filtered["control_mode"] == "control"]
    filtered = filtered[filtered["trial"] == trial_idx]

    start_idx, stop_idx = 0, 0

    found = False
    # just look for a trajectory that is inbound/outbound phase, because
    # if you look for a trajectory center to right in an inbound phase it
    # could first go to center, then to the left, and then to the right
    while not found:
        filtered = filtered[filtered["time"] > start_idx]

        # find start index
        fi = filtered[filtered["pos_x"] == from_x]
        fi = fi[fi["pos_y"] == 1]
        start_idx = fi["time"].to_numpy()[0]

        # find stop index
        fi = filtered[filtered["time"] > start_idx]
        fi = fi[fi["pos_x"] == to_x]
        fi = fi[fi["pos_y"] == 1]
        stop_idx = fi["time"].to_numpy()[0]

        fi = filtered[filtered["time"] >= start_idx]
        fi = fi[fi["time"] <= stop_idx]

        ib = fi["inbound"].to_numpy()
        found = np.all(ib[:-1] == ib[0])

    x, y = fi["pos_x"].to_numpy().astype(np.float32), fi[
        "pos_y"
    ].to_numpy().astype(np.float32)
    # x += np.random.randn(*x.shape) * 0.075
    # y += np.random.randn(*x.shape) * 0.075
    c = plt.get_cmap("Reds")(0.33 + 0.75 * (np.arange(len(x)) / len(x)))
    for i, (x1, x2, y1, y2) in enumerate(zip(x[:-1], x[1:], y[:-1], y[1:])):
        ax.plot(
            [x1, x2],
            [y1, y2],
            c=c[i],
            alpha=0.33 + i / (2 * len(x)),
            linestyle="--",
        )
