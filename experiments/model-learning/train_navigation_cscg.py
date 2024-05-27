import torch

import numpy as np
import matplotlib.pyplot as plt

from hippo import get_store_path, get_data_path
from hippo import save_config
from hippo import load_config, get_recents_path
from hippo.data import load_data
from hippo.visualization import format_ax
from hippo.models.cscg import CHMM
import pickle
import igraph

from matplotlib import cm

import logging

logger = logging.getLogger(__name__)


def get_graph(chmm, x, a, multiple_episodes=False):
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A.astype(np.float16) > 1e-4).tolist())
    return g, v


def plot_graph(
    chmm,
    x,
    a,
    output_file,
    n_uncloned_states,
    cmap=cm.Spectral,
    multiple_episodes=False,
    vertex_size=30,
):
    g, v = get_graph(chmm, x, a, multiple_episodes)

    node_labels = np.arange(n_uncloned_states).repeat(n_clones)[v]
    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    nc = 20
    recent = False

    if not recent:
        data_path = get_data_path() / "train_data/full_exploration.npz"
    else:
        data_path = load_config(get_recents_path() / "dataset_path.yml")[
            "dataset_path"
        ]

    logger.info(f"Loaded data from {data_path}")

    store_path = get_store_path("train-CSCG")
    logger.info(f"Storing results at {store_path}")

    save_config(
        {"cscg_path": str(store_path)}, get_recents_path() / "cscg_path.yml"
    )

    states, actions, _, tokenizer = load_data(
        data_path,
        drop_last=False,
        block_size=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    logging.info("Loaded data")

    states = states.astype(np.int64)
    actions = actions.astype(np.int64)

    n_obs = len(tokenizer.codebook)

    n_clones = nc * np.ones(n_obs, dtype=np.int64)

    logging.info(f"Creating a CSCG with {n_clones} clones")

    chmm = CHMM(n_clones=n_clones, x=states, a=actions, pseudocount=1e-10)

    progression = chmm.learn_em_T(
        states, actions, n_iter=10000, term_early=True
    )

    fig, ax = plt.subplots(1, 1, figsize=(2, 1))
    format_ax(ax)
    ax.plot(progression, color="tab:red")
    plt.savefig(store_path / "train_progression.png", bbox_inches="tight")
    plt.clf()

    # refine learning
    chmm.pseudocount = 0.0
    _ = chmm.learn_viterbi_T(states, actions, n_iter=100)

    with open(store_path / "chmm.pkl", "wb") as outp:
        pickle.dump(chmm, outp, pickle.HIGHEST_PROTOCOL)

    with open(store_path / "tokenizer.pkl", "wb") as outp:
        pickle.dump(tokenizer, outp, pickle.HIGHEST_PROTOCOL)

    plot_graph(
        chmm,
        states,
        actions,
        str(store_path / "out.pdf"),
        n_uncloned_states=len(tokenizer.codebook),
    )
