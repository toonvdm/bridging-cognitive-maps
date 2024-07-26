import numpy as np

from itertools import product
from hippo.models.cscg_tools import extract_AB

from pymdp.agent import Agent
from pymdp.utils import obj_array_from_list

from pymdp.control import (
    get_expected_states_interactions,
    get_expected_obs_factorized,
    calc_expected_utility,
    calc_states_info_gain_factorized,
)
from gymnasium import spaces

import logging

logger = logging.getLogger(__name__)


class FakeTokenizer:
    def __init__(self):
        # Hippo state index to corridor mapping
        self._state_to_hippo = [3, 4, 2]

    def hippo_to_state(self, x):
        if x in self._state_to_hippo:
            return self._state_to_hippo.index(x)
        else:
            return None

    def state_to_hippo(self, state):
        return self._state_to_hippo[state]


class LocationRewardMerger:
    def __init__(self):
        # location, reward
        self._obs_to_state = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    def obs_to_state(self, loc, reward):
        return 2 * loc + reward

    def state_to_hippo(self, state):
        return self._obs_to_state[state]


class LocationPrevRewardMerger:
    def __init__(self):
        # Observe the location where the previous reward was
        # collected
        # (location, reward, previous_location)
        self._obs_to_state = [
            (2, 0, 2),
            (1, 0, 1),
            (0, 1, 0),
            (0, 0, 0),
            (2, 0, 1),
            (2, 1, 2),
            (0, 0, 2),
            (2, 0, 0),
            (0, 0, 1),
            (1, 1, 1),
        ]

    def obs_to_state(self, loc, reward, prev_loc):
        return self._obs_to_state.index((loc, reward, prev_loc))

    def state_to_hippo(self, state):
        return self._obs_to_state[state]


class PrefrontalFactorizedModel:
    def __init__(
        self, rewards, n_rewards, n_steps, n_rules, n_locations, n_actions
    ):
        a_reward = np.zeros((n_rewards, n_steps, n_rules, n_locations))
        for step in range(n_steps):
            for rule in range(n_rules):
                for location in range(n_locations):
                    reward = int((step, rule, location) in rewards)
                    # the reward arrives at the next step if you take it
                    a_reward[reward, (step + 1) % n_steps, rule, location] = (
                        1.0
                    )

        a_location = np.eye(3)

        b_step = np.zeros((n_steps, n_steps, n_rules, n_actions))
        for step in range(n_steps):
            for rule in range(n_rules):
                for action in range(n_actions):
                    # step rule loc
                    next_step = step
                    next_step += int((step, rule, action) in rewards)
                    next_step %= n_steps
                    b_step[next_step, step, rule, action] = 1.0

        b_rule = np.eye(n_rules)
        b_rule = b_rule.reshape(*b_rule.shape, 1)
        b_rule = b_rule.repeat(n_actions, -1)

        b_location = np.zeros((n_locations, n_locations, n_actions))
        for action in range(n_actions):
            b_location[action, :, action] = 1.0

        A = [a_reward, a_location]
        A = [a / a.sum(axis=0, keepdims=True) for a in A]
        A = obj_array_from_list(A)
        B = [b_step, b_rule, b_location]
        B = [b / b.sum(axis=0, keepdims=True) for b in B]
        B = obj_array_from_list(B)

        # Likelihood:
        # reward is conditioned on: step, rule location, action
        # location is conditioned on: location, action
        A_factor_list = [[0, 1, 2], [2]]
        # Transition:
        # Step is conditioned on step and rule
        # Rule is conditioned on rule
        # Location is conditioned on location
        B_factor_list = [[0, 1], [1], [2]]

        C = obj_array_from_list([np.array([0.0, 1.0]), np.ones(3)])

        self.C = C

        policies = []
        for i in range(n_actions):
            policy = i * np.ones((1, len(B)), dtype=int)
            policies += [policy]
        policies = obj_array_from_list(policies)

        self.agent = Agent(
            A,
            B,
            C,
            A_factor_list=A_factor_list,
            B_factor_list=B_factor_list,
            policy_len=1,
            policies=policies,
            action_selection="stochastic",
        )

        self.tokenizer = FakeTokenizer()
        self.prev_action = 0

    def reset(self):
        self.agent.reset()
        self.prev_action = 0

    def get_hippo_empirical_prior(self):
        return self.tokenizer.state_to_hippo(self.prev_action)

    def _eval_policy(self, policy):
        self.pomdp = self.agent
        qs_pi = get_expected_states_interactions(
            self.pomdp.qs, self.pomdp.B, self.pomdp.B_factor_list, policy
        )
        qo_pi = get_expected_obs_factorized(
            qs_pi, self.pomdp.A, self.pomdp.A_factor_list
        )
        e_u = calc_expected_utility(qo_pi, self.pomdp.C)
        e_ig = calc_states_info_gain_factorized(
            self.pomdp.A, qs_pi, self.pomdp.A_factor_list
        )
        EFE = e_u + e_ig

        return qs_pi, qo_pi, e_u, e_ig, EFE

    def observation_likelihood(self, state, observation_factor=0):
        return self.agent.A[observation_factor][:, state].argmax()

    def _debug_efe(self):
        utility, infogain = np.zeros(3), np.zeros(3)
        for i, policy in enumerate(self.agent.policies):
            _, _, e_u, eig, _ = self._eval_policy(policy)
            utility[i] = e_u
            infogain[i] = eig
        return utility, infogain

    def act(self, obs, reward):
        loc = self.tokenizer.hippo_to_state(obs)
        if loc is not None:
            # agent has not yet reached a sensible location
            # bootstrap
            self.agent.infer_states([reward, loc])
            # Inhibition
            C_1 = np.ones_like(self.C[1])
            C_1[loc] = -10
            self.agent.C[1] = C_1

        q_pi, efe = self.agent.infer_policies_factorized()

        # compute the values for logging
        utility, infogain = self._debug_efe()

        action = self.agent.sample_action()
        self.prev_action = int(action[0])
        return action, {
            "lower_state": self.get_hippo_empirical_prior(),
            "qs": self.agent.qs.copy(),
            "q_pi": q_pi.copy(),
            "utility": utility,
            "infogain": infogain,
        }


class PrefrontalCSCGModel:
    def __init__(
        self,
        cscg,
        with_location=False,
        with_light=False,
        flat_preference=False,
    ):
        self.tokenizer = FakeTokenizer()
        self.cscg = cscg
        A, B = extract_AB(cscg, reduce=True)

        self.with_location = with_location
        self.with_light = with_light

        self.obs_merger = None
        if self.with_location:
            C = np.array([0, 5, 0, 5, 0, 5, -1])
            self.obs_merger = LocationRewardMerger()
        elif self.with_light:
            self.obs_merger = LocationPrevRewardMerger()
            C = np.array([k[1] for k in self.obs_merger._obs_to_state] + [-1])
        else:
            C = np.array([0, 5, -1])

        if flat_preference:
            C = np.zeros_like(C)

        gamma = 0.5
        self.agent = Agent(
            A=A,
            B=B,
            C=C,
            gamma=gamma,
            action_selection="stochastic",
        )
        self.agent.policies = self.construct_policies(1)
        self.prev_action = 0

        self.pE = np.ones_like(self.agent.E)

    def add_pE(self, a, s):
        self.pE = np.ones((3, a.max() + 1)).astype(np.float32)
        for ai, si, sn in zip(a[:-1], s[:-1], s[1:]):
            # action and o taken at the current step
            o, _ = self.obs_merger.state_to_hippo(si)[:2]
            # Reward at the next step
            r = self.obs_merger.state_to_hippo(sn)[1]
            self.pE[o][ai] += r

    def construct_policies(self, n):
        policies = np.array(list(product([0, 1, 2], repeat=n)))
        policies = policies.reshape((-1, n, 1))
        return policies

    def reset(self):
        self.agent.reset()
        self.prev_action = 0

    def observation_likelihood(self, state):
        return self.agent.A[0][:, state].argmax()

    def get_hippo_empirical_prior(self):
        return self.tokenizer.state_to_hippo(self.prev_action)

    def set_E(self, cond):
        self.agent.E = self.pE[cond] / self.pE[cond].sum()

    def act(self, obs, reward=None, prev=None):
        # logger.info(f"Reward: {reward}")
        if self.with_location:
            # First get the hippo observation for loc
            loc = self.tokenizer.hippo_to_state(obs)
            if loc is not None:
                obs = self.obs_merger.obs_to_state(loc, reward)
                self.agent.infer_states([obs])
                self.set_E(loc)
        elif self.with_light:
            # First get the hippo observation for loc
            loc = self.tokenizer.hippo_to_state(obs)
            if loc is not None:
                obs = self.obs_merger.obs_to_state(loc, reward, prev)
                self.agent.infer_states([obs])
        else:
            self.agent.infer_states([reward])

        q_pi, efe = self.agent.infer_policies()
        action = self.agent.sample_action()
        self.prev_action = int(action[0])

        return action, {
            "lower_state": self.get_hippo_empirical_prior(),
            "qs": self.agent.qs.copy(),
            "q_pi": q_pi.copy(),
        }


class PrefrontalRandomModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.sample_space = spaces.Discrete(3)

    def reset(self):
        self.prev_action = 0

    def observation_likelihood(self, state):
        return -1

    def get_hippo_empirical_prior(self):
        return self.tokenizer.state_to_hippo(self.prev_action)

    def act(self, obs, reward):
        self.prev_action = self.sample_space.sample()
        return self.prev_action, {
            "lower_state": self.get_hippo_empirical_prior()
        }
