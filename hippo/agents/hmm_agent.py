import numpy as np

import torch

from hippo.models.hmm import HMM
from hippo.agents.memory import Memory
from hippo.free_energy.tree import construct_tree
from hippo.models.cscg_tools import add_death
from hippo.free_energy import free_energy

from pymdp.maths import softmax

from gymnasium import spaces

import networkx as nx

import logging


logger = logging.getLogger(__name__)


def extract_AB_from_hmm(hmm: HMM):
    death = 1
    n_obs = 19 + death
    n_states = hmm.transition.shape[1] + death

    T = np.array(hmm.transition)
    T = T.transpose(2, 1, 0)
    T *= T > 1e-3

    v = T.sum(axis=0).sum(axis=1).nonzero()[0]
    T = T[v, :][:, v]

    # B
    B = np.zeros((T.shape[0] + death, T.shape[0] + death, T.shape[2]))
    B[: T.shape[0], : T.shape[1]] = T
    B = add_death(B)
    B /= B.sum(axis=0, keepdims=True)

    A = np.zeros((n_obs, n_states))
    A[:-1, :-1] = np.array(hmm.emission).T
    A[-1, -1] = 1.0

    v = np.concatenate([v, np.array([-1])])
    A = A[:, v]

    # Normalize over state: Sum_s P(o|s) = 1
    A /= A.sum(axis=0, keepdims=True)

    return A, B


class HMMAgent:
    def __init__(
        self,
        hmm,
        tokenizer,
        episode_length=100,
        n_obs=None,
        n_actions=4,
        inference_length=100,
        policy_len=2,
        random_actions=False,
        preference_diffusion=True,
        gamma=16.0,
        death_state=True,
        reduce=True,
    ):
        self.tokenizer = tokenizer
        self.death_state = death_state
        self.reduce = reduce

        self.memory = Memory()
        self.episode_length = episode_length
        self.inference_length = inference_length

        # Do setting of these variables in the reset method
        self.n_steps = 0
        self.actions = None
        self.observations = None

        # some other useful variables
        self.policy_len = policy_len
        self.n_actions = n_actions
        self.n_obs = n_obs
        if self.n_obs is None:
            self.n_obs = len(self.tokenizer.codebook)

        self.gamma = gamma  # temperature for action sampling

        self.use_preference_diffusion = preference_diffusion
        self.random_actions = random_actions

        self.hmm = None
        self.T = None
        self.likelihood = None
        self.set_hmm(hmm)

        # do not train on the stationary action
        self.action_space = spaces.Discrete(self.n_actions - 1)

        self.constraint = np.zeros(self.T.shape[1]) + 1e-6  # C

        self.previous_states = []

        self.reset()

    def set_hmm(self, hmm):
        self.hmm = hmm
        likelihood, transition = extract_AB_from_hmm(self.hmm)
        transition = transition.transpose(2, 0, 1)

        self.likelihood = likelihood
        self.T = transition

    def reset(self):
        self.end_episode()

        # clear the cached poperties
        # for k in ["T", "v", "state_loc", "likelihood"]:
        for k in ["v", "state_loc"]:
            if k in self.__dict__:
                del self.__dict__[k]

        # uniform prior
        self.qs = np.ones(self.T.shape[1])
        self.qs /= self.qs.sum()

    def update_states(self, qs):
        self.qs = qs

    def update_obs(self, obs):
        self.observations[self.n_steps] = obs

    def update_action(self, action):
        self.actions[self.n_steps] = action

    @property
    def T_tr(self):
        # shape: action, to, from
        return self.T.transpose(0, 2, 1)

    def transition(self, belief, action):
        q_next = self.T[action].dot(belief) + 1e-12
        q_next = q_next / q_next.sum()
        return q_next

    def set_state_preference(self, preferred_state):
        max_val = 15

        # States that are not connected should have a very low value
        self.constraint = np.zeros(self.T.shape[1])

        if self.use_preference_diffusion:
            adjacency = self.T.sum(axis=0)[:-1, :-1] > 1e-4
            graph = nx.from_numpy_array(adjacency.T, create_using=nx.DiGraph)
            p = nx.shortest_path_length(graph, target=preferred_state)
            for k, v in p.items():
                self.constraint[k] = max_val - v

        # Unless its the preference
        self.constraint[preferred_state] = max_val

        # worse thant the worst move
        self.constraint[-1] = self.constraint[:-1].min() - 1

    def inhibit(self, qs):
        C = self.constraint.copy()

        preferred_state = self.constraint.argmax()

        state = qs.argmax()
        if state != preferred_state:
            C[state] = 0
        return C

    def end_episode(self):
        # Keep the episode in memory and reset for new capture
        if self.n_steps > 0:
            self.memory.add_episode(
                self.observations[: self.n_steps],
                self.actions[: self.n_steps],
            )
        self.n_steps = 0
        self.actions = np.zeros((self.episode_length), dtype=np.int64)
        self.observations = np.zeros((self.episode_length), dtype=np.int64)

    def step_time(self):
        self.n_steps += 1
        if self.n_steps > self.episode_length:
            raise NotImplementedError("Fix rolling window")

    def infer_states(self, obs, qs_prev, aij=None):
        """
        Compute the posterior belief over state, given the observation.
        """
        if self.n_steps < 1:
            j = obs
            # Step 0 -> just use the prior
            qs = self.likelihood[j, :]
            prior = qs.copy()
        else:
            if aij is None:
                aij = self.actions[self.n_steps - 1]

            prior = self.T[aij].dot(qs_prev)
            likelihood = self.likelihood[obs, :]
            qs = prior * likelihood

            if qs.sum() == 0:
                o = self.likelihood[:, qs_prev.argmax()].argmax()
                print(o, aij)

        qs /= qs.sum()
        return qs

    def infer_action(self, qs):
        if self.random_actions:
            # Do not sample the rest action, as we can just force it to be
            # identity matrix
            action = self.action_space.sample()
            selection_data = {}
        else:
            # Inhibit both current and previous state
            C = self.inhibit(qs)

            tree = construct_tree(self, qs, C)
            nefe = np.array([p[-1].data["nefe"] for p in tree])

            q_pi = softmax(self.gamma * nefe)
            policies = np.array([[[pi.action] for pi in p] for p in tree])

            selected_idx = np.random.choice(np.arange(len(policies)), p=q_pi)
            # selected_idx = nefe.argmax()
            action = policies[selected_idx][0][0]

            selection_data = {
                "tree": tree,
                "selected": selected_idx,
                "constraint": C,
                "q_pi": q_pi,
            }

        return action, selection_data

    def act(self, pixel_obs, reward=0):
        obs = self.tokenizer(pixel_obs.reshape(-1, *pixel_obs.shape))[0]
        qs = self.infer_states(obs, self.qs)
        action, efe_logs = self.infer_action(qs)

        # update values and step time
        self.update_states(qs)
        self.update_obs(obs)
        self.update_action(action)
        self.step_time()

        fe = free_energy(self.qs, self.constraint, None)[0]
        return action, {
            "efe": efe_logs,
            "qs_clones": qs,
            "state": qs.argmax(),
            "free_energy": fe,
        }

    def replay_step(self, pixel_obs, action):
        obs = self.tokenizer(pixel_obs.reshape(-1, *pixel_obs.shape))[0]
        qs = self.infer_states(obs, self.qs)
        self.update_states(qs)
        self.update_obs(obs)

        self.update_action(action)
        self.step_time()
        return qs
