import logging
import numpy as np

from pathlib import Path

from hippo import get_data_path, get_store_path, save_config
from hippo.agents.clone_agent import CloneAgent
from hippo.evaluation.replay import ReplayEvaluation
from hippo.evaluation.planning import PlannerEvaluation
from hippo import save_object, load_object, load_config, get_recents_path

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store_path = get_store_path("navigation-clone-agent")

    cscg_path = get_data_path() / "models/navigation"
    agent_path = cscg_path / "cscg_hippo.pkl"
    tokenizer_path = cscg_path / "tokenizer.pkl"

    config = {
        "cscg_path": str(agent_path),
        "tokenizer_path": str(tokenizer_path),
        "agent": {
            "reduce": True,
            "gamma": 0.5,
            "policy_len": 5,
            "random_actions": False,
            "preference_diffusion": True,
        },
    }

    tokenizer = load_object(config["tokenizer_path"])

    fig, ax = plt.subplots(4, 5, figsize=(10, 8))
    for i, a in enumerate(ax.flatten()):
        if i < len(tokenizer.codebook):
            o = tokenizer.get_observation(i).astype(np.float32)
            a.imshow(o / 255.0)
            a.set_title(i)

    [a.axis("off") for a in ax.flatten()]
    plt.savefig(store_path / "observations.png", bbox_inches="tight")
    plt.close()

    cscg = load_object(config["cscg_path"])
    print(cscg)

    agent = CloneAgent(
        tokenizer=tokenizer,
        cscg=cscg,
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
    save_object(cscg, store_path / "chmm.pkl")

    planning_evaluator = PlannerEvaluation(store_path, debug_sequences=False)
    planning_evaluator.evaluate_agent(agent)

    inference_evaluator = ReplayEvaluation(store_path)
    inference_evaluator.evaluate_agent(agent, run_index=0)
