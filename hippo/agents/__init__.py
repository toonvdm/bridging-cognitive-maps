from hippo import load_object, save_object, get_data_path
from hippo.agents.clone_agent import CloneAgent
from hippo.agents.replay_agent import ReplayAgent
from hippo.agents.hierarchical_agent import HierarchicalAgent
from hippo.agents.prefrontal_agent import (
    PrefrontalCSCGModel,
    PrefrontalFactorizedModel,
    PrefrontalRandomModel,
)

from pathlib import Path

from enum import StrEnum


class AgentModes(StrEnum):
    FACTORIZED = "FACTORIZED"
    CSCG_NOLOC = "CSCG_NOLOC"
    CSCG_LOC = "CSCG_LOC"
    CSCG_LOC_RULE0 = "CSCG_LOC_RULE0"
    CSCG_LIGHT_RULE0 = "CSCG_LIGHT_RULE0"
    RANDOM = "RANDOM"


__all__ = [
    "CloneAgent",
    "ReplayAgent",
    "HierarchicalAgent",
]


def load_agent(load_path):
    return load_object(load_path)


def save_agent(obj, load_path):
    return save_object(obj, load_path)


def load_hippo_agent(policy_len=5):
    """
    Shortcut for loading the hippo agent developed in the navigation experiments
    """
    cscg_path = get_data_path() / "models/navigation"
    agent_path = cscg_path / "cscg_hippo.pkl"
    tokenizer_path = cscg_path / "tokenizer.pkl"

    config = {
        "cscg_path": str(agent_path),
        "tokenizer_path": str(tokenizer_path),
        "agent": {
            "reduce": True,
            "gamma": 0.5,
            "policy_len": policy_len,
            "random_actions": False,
            "preference_diffusion": True,
        },
    }

    tokenizer = load_object(config["tokenizer_path"])
    cscg = load_object(config["cscg_path"])

    agent = CloneAgent(
        tokenizer=tokenizer,
        cscg=cscg,
        random_actions=config["agent"]["random_actions"],
        policy_len=config["agent"]["policy_len"],
        reduce=config["agent"]["reduce"],
        gamma=config["agent"]["gamma"],
        preference_diffusion=config["agent"]["preference_diffusion"],
        episode_length=1801,
    )
    agent.reset()
    return agent


def load_prefrontal_agent(agent_mode=AgentModes.FACTORIZED):
    """
    Shortcut for loading the prefrontal agent developed in the navigation experiments
    factorized = True: factorized AIF model
    factorized = False: CSCG model
    """
    cscg_path = get_data_path() / "models/task"

    if agent_mode.value == AgentModes.FACTORIZED.value:
        # Manually designed model
        n_rewards = 2
        n_steps_in_rule = 4
        n_rules = 3
        n_locations = 3
        n_actions = n_locations

        rewards = [
            # alternation rule LCRL
            (0, 0, 0),  # Step 0, Rule 0, Left
            (1, 0, 1),  # Step 1, Rule 0, Center
            (2, 0, 2),  # Step 2, Rule 0, Right
            (3, 0, 1),  # Step 3, Rule 0, Center
            # alternation rule LCLR
            (0, 1, 0),  # Step 0, Rule 1, Left
            (1, 1, 1),  # Step 1, Rule 1, Center
            (2, 1, 0),  # Step 2, Rule 1, Left
            (3, 1, 2),  # Step 3, Rule 1, Right
            # alternation rule RCRL
            (0, 2, 2),  # Step 0, Rule 1, Right
            (1, 2, 1),  # Step 1, Rule 1, Center
            (2, 2, 2),  # Step 2, Rule 1, Right
            (3, 2, 0),  # Step 3, Rule 1, Left
        ]

        model_pf = PrefrontalFactorizedModel(
            rewards,
            n_rewards,
            n_steps_in_rule,
            n_rules,
            n_locations,
            n_actions,
        )
        # Flat habit for this agent

    elif agent_mode.value == AgentModes.CSCG_NOLOC.value:
        cscg_path /= "cscg_prefrontal_3_rules.pkl"
        pf_cscg = load_object(cscg_path)
        model_pf = PrefrontalCSCGModel(pf_cscg, with_location=False)
        # Flat habit for this agent

    elif agent_mode.value == AgentModes.CSCG_LOC.value:
        cscg_path /= "cscg_prefrontal_with_loc_3_rules.pkl"
        pf_cscg = load_object(cscg_path)
        model_pf = PrefrontalCSCGModel(pf_cscg, with_location=True)

        a = load_object(get_data_path() / "train_data/actions_3_rules.pkl")
        s = load_object(get_data_path() / "train_data/states_3_rules.pkl")
        model_pf.add_pE(a=a, s=s)

    elif agent_mode.value == AgentModes.CSCG_LOC_RULE0.value:
        cscg_path /= "cscg_prefrontal_with_loc_rule_0.pkl"
        pf_cscg = load_object(cscg_path)
        model_pf = PrefrontalCSCGModel(pf_cscg, with_location=True)

        a = load_object(get_data_path() / "train_data/actions_rule_0.pkl")
        s = load_object(get_data_path() / "train_data/states_rule_0.pkl")
        model_pf.add_pE(a=a, s=s)

    elif agent_mode.value == AgentModes.CSCG_LIGHT_RULE0.value:
        cscg_path /= "cscg_prefrontal_with_light_rule0.pkl"
        pf_cscg = load_object(cscg_path)
        model_pf = PrefrontalCSCGModel(pf_cscg, with_light=True)
        # Flat habit for this agent

    elif agent_mode.value == AgentModes.RANDOM.value:
        model_pf = PrefrontalRandomModel()
    else:
        raise NotImplementedError

    return model_pf
