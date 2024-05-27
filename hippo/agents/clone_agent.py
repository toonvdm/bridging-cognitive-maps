import numpy as np

from hippo.models.cscg_tools import extract_AB
from hippo.agents.memory import Memory
from hippo.free_energy.tree import construct_tree
from hippo.free_energy import free_energy

from pymdp.maths import softmax

from gymnasium import spaces

import networkx as nx
from functools import cached_property


class CloneAgent:
    def __init__(
        self,
        cscg,
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
        epistemic=True,
    ):

        self.use_epistemic = epistemic

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

        self.cscg = None
        self.T = None
        self.likelihood = None
        self.set_cscg(cscg)

        # do not train on the stationary action
        self.action_space = spaces.Discrete(self.n_actions - 1)

        self.constraint = np.zeros(self.T.shape[1]) + 1e-6  # C

        self.preference_max_val = 15

        self.previous_states = []

        self.reset()

    def set_cscg(self, cscg):
        self.cscg = cscg
        likelihood, transition = extract_AB(
            self.cscg, self.reduce, False, self.death_state
        )
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
        self.qs = np.ones(self.v.shape)
        self.qs /= self.qs.sum()

    def update_states(self, qs):
        self.qs = qs

    def update_obs(self, obs):
        self.observations[self.n_steps] = obs

    def update_action(self, action):
        self.actions[self.n_steps] = action

    @cached_property
    def v(self):
        if self.reduce:
            # Sum over actions and from state to only get relevant states
            states = self.cscg.T.sum(axis=0).sum(axis=0) > 1e-4
            v = states.nonzero()[0]
        else:
            v = np.arange(self.cscg.T.shape[1])
        if self.death_state:
            v = np.concatenate([v, [self.cscg.T.shape[1]]])
        return v

    @cached_property
    def state_loc(self):
        state_loc = np.hstack(
            (np.array([0], dtype=self.cscg.n_clones.dtype), self.cscg.n_clones)
        ).cumsum()
        state_loc_new = [0]
        for start, end in zip(state_loc[:-1], state_loc[1:]):
            res = np.where(np.logical_and(self.v >= start, self.v < end))[0]
            state_loc_new.append(state_loc_new[-1] + len(res))

        return np.array(state_loc_new)

    @property
    def T_tr(self):
        # shape: action, to, from
        return self.T.transpose(0, 2, 1)

    def transition(self, belief, action):
        q_next = self.T[action].dot(belief) + 1e-12
        q_next = q_next / q_next.sum()
        return q_next

    def set_state_preference(self, preferred_state):
        max_val = self.preference_max_val

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
        self.constraint = self.constraint

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

    def infer_states_from_sequence(self, episode=None):
        if episode is None:
            end = self.n_steps + 1
            start = max(end - self.inference_length, 0)
            obs = self.observations[start:end]
            acs = self.actions[start:end]
        else:
            obs, acs = episode

        qs = self.cscg.infer_states(obs, acs)
        qs /= qs.sum()
        return qs

    def infer_states(self, obs, qs_prev, aij=None):
        """
        Compute the posterior belief over state, given the observation.
        """
        if self.n_steps < 1:
            j = obs
            j_start, j_stop = self.state_loc[j : j + 2]
            # Step 0 -> just use the prior
            qs = np.zeros_like(qs_prev)
            qs[j_start:j_stop] = self.cscg.Pi_x[j_start:j_stop]
            prior = qs.copy()
        else:
            if aij is None:
                aij = self.actions[self.n_steps - 1]

            prior = self.T[aij].dot(qs_prev)
            likelihood = self.likelihood[obs, :]
            qs = prior * likelihood + 1e-4

            if qs.sum() == 0:
                o = self.likelihood[:, qs_prev.argmax()].argmax()
                print(o, aij)

        qs /= qs.sum() + 1e-4
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

            tree = construct_tree(self, qs, C, epistemic=self.use_epistemic)
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
        j_start, j_stop = self.state_loc[obs : obs + 2]
        return action, {
            "efe": efe_logs,
            "qs_clones": qs[j_start:j_stop],
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
