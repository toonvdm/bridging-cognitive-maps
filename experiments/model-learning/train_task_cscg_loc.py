import numpy as np
import logging

import random

from hippo.environments.w_maze import WmazeTasks

from hippo import get_store_path, save_config, save_object
from hippo.agents.prefrontal_agent import LocationRewardMerger
from hippo.models.cscg import CHMM

from hippo.environments.high_level import HighLevelWMazeEnvironment

logger = logging.getLogger(__name__)


def generate_sequence(
    p_optim=0.0,
    sequence_length=2500,
    cycle_rule_every=1000,
    added_task_size=None,
    n_rules=3,
):
    tokenizer = LocationRewardMerger()

    env = HighLevelWMazeEnvironment(
        cycle_rule_every, added_task_size, n_rules
    )  # 500
    actions = np.random.randint(0, 3, size=sequence_length, dtype=np.int64)
    rewards = np.zeros_like(actions, dtype=np.int64)
    locations = np.zeros_like(actions, dtype=np.int64)
    for i in range(actions.shape[0] - 1):
        # With certain probability choose_optimal action
        if np.random.random() < p_optim:
            actions[i] = env.next_location
        locations[i + 1] = actions[i]
        rewards[i + 1] = env.step(actions[i])
    states = tokenizer.obs_to_state(locations, rewards)
    return states, actions


def fit_cscg(states, actions, n_clones):
    logging.info(f"Creating a CSCG with {n_clones} clones")
    chmm = CHMM(n_clones=n_clones, x=states, a=actions, pseudocount=1e-10)
    chmm.learn_em_T(states, actions, n_iter=10000, term_early=True)
    # Refine learning
    chmm.pseudocount = 0.0
    chmm.learn_viterbi_T(states, actions, n_iter=100)
    return chmm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store_path = get_store_path("train-prefrontal-cscg-with-location")
    logger.info(f"Storing results at {store_path}")

    n_rules = 1
    rule_tag = "3_rules" if n_rules == 3 else "rule_0"

    config = {
        "n_clones": 10,
        "seed": 0,
        "sequence_length": 2000 * 4,
        "cycle_rule_every": 1000,
        "p_optim": 0.75,
    }

    np.random.seed(config["seed"])
    random.seed(config["seed"])

    states, actions = generate_sequence(
        sequence_length=config["sequence_length"],
        p_optim=config["p_optim"],
        cycle_rule_every=config["cycle_rule_every"],
        n_rules=n_rules,
    )
    print(states, actions)

    logger.info(f"Generated data of shape: {states.shape}")

    n_obs = len(np.unique(states))
    logger.info(f"{n_obs} observations")
    n_clones = config["n_clones"] * np.ones(n_obs, dtype=np.int64)
    cscg = fit_cscg(states, actions, n_clones)
    logger.info("CSCG Fitted")

    save_object(actions, store_path / f"actions_{rule_tag}.pkl")
    save_object(states, store_path / f"states_{rule_tag}.pkl")
    save_object(cscg, store_path / f"chmm_prefrontal_with_loc_{rule_tag}.pkl")
    save_config(
        config, store_path / f"config_chmm_prefrontal_with_loc_{rule_tag}.yml"
    )
