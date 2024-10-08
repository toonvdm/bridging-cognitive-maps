import logging
import numpy as np

from pathlib import Path

from hippo import get_store_path, save_config
from hippo.agents.hmm_agent import HMMAgent
from hippo.evaluation.replay import ReplayEvaluation
from hippo.evaluation.planning import PlannerEvaluation
from hippo import save_object, load_object, get_data_path

from hippo.models.hmm import HMM

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store_path = get_store_path("navigation-hmm-agent-viterbi")

    hmm_path = get_data_path() / "models/navigation"
    agent_path = hmm_path / "hmm_hippo.pkl"
    tokenizer_path = hmm_path / "hmm_tokenizer.pkl"

    config = {
        "hmm_path": str(agent_path),
        "tokenizer_path": str(tokenizer_path),
        "agent": {
            "reduce": True,
            "gamma": 16,
            "policy_len": 5,
            "random_actions": False,
            "preference_diffusion": True,
        },
    }

    tokenizer = load_object(config["tokenizer_path"])

    fig, ax = plt.subplots(4, 5, figsize=(10, 8))
    for i, a in enumerate(ax.flatten()):
        if i < len(tokenizer.codebook):
            obs = np.array(tokenizer.get_observation(i).tolist()) / 255.0
            a.imshow(obs)
            a.set_title(f"{i}")
    [a.axis("off") for a in ax.flatten()]
    plt.savefig(store_path / "observations.png", bbox_inches="tight")
    plt.close()

    hmm = load_object(config["hmm_path"])

    agent = HMMAgent(
        tokenizer=tokenizer,
        hmm=hmm,
        random_actions=config["agent"]["random_actions"],
        policy_len=config["agent"]["policy_len"],
        reduce=config["agent"]["reduce"],
        gamma=config["agent"]["gamma"],
        preference_diffusion=config["agent"]["preference_diffusion"],
    )
    agent.reset()

    # Save everything
    save_config(config, store_path / "config.yml")
    save_object(agent, store_path / "agent.pkl")
    save_object(tokenizer, store_path / "tokenizer.pkl")
    save_object(hmm, store_path / "hmm.pkl")

    planning_evaluator = PlannerEvaluation(store_path, debug_sequences=False)
    planning_evaluator.evaluate_agent(agent)

    inference_evaluator = ReplayEvaluation(store_path)
    inference_evaluator.evaluate_agent(agent, run_index=0)
