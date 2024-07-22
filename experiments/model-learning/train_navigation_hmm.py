import jax.numpy as jnp

from hippo import get_store_path, get_data_path
from hippo import save_config
from hippo import load_config, get_recents_path
from hippo.data import load_data
from pathlib import Path
import pickle

import logging

from hippo.models.hmm import HMM

logger = logging.getLogger(__name__)


def print_params(params):
    jnp.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    print("initial probs:")
    print(params.initial.probs.shape)
    print("transition matrix:")
    print(params.transitions.transition_matrix.shape)
    print("emission probs:")
    print(params.emissions.probs[:, 0, :].shape)  # since num_emissions = 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Whether to use the recent file to get the data
    recent = False
    if not recent:
        data_path = get_data_path() / "train_data/full_exploration.npz"
    else:
        data_path = load_config(get_recents_path() / "dataset_path.yml")[
            "dataset_path"
        ]

    logger.info(f"Loaded data from {data_path}")

    # If training crashes, rerun from a checkpoint
    store_path = get_store_path("train-HMM")
    start_from = None

    logger.info(f"Storing results at {store_path}")
    save_config(
        {"hmm_path": str(store_path)}, get_recents_path() / "hmm_path.yml"
    )

    states, actions, _, tokenizer = load_data(
        data_path, drop_last=False, block_size=1000, device="cpu"
    )
    logging.info("Loaded data")

    n_train = len(actions)
    observations, actions = (
        jnp.array(states)[:n_train],
        jnp.array(actions)[:n_train],
    )

    n_obs = len(tokenizer.codebook)

    nc = 20

    with open(store_path / "tokenizer.pkl", "wb") as outp:
        pickle.dump(tokenizer, outp, pickle.HIGHEST_PROTOCOL)

    hmm = HMM(
        n_observations=n_obs,
        n_states=nc * n_obs,
        n_actions=3,
        store_path=store_path,
    )
    hmm.fit_baum_welch(
        observations,
        actions,
        save_every=10,
        n_iters=1000,
        start_from=start_from,
    )
