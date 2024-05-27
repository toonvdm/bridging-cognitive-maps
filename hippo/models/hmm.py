from tqdm import tqdm

import copy

import jax
import jax.numpy as jnp
import numpy as np

from tqdm import tqdm
import scipy.stats as ss

from collections import namedtuple

import pickle
import logging

from hippo.models.cscg import rargmax

logger = logging.getLogger(__name__)

eps = 1e-14

HMMParams = namedtuple(
    "HMMParams", ("transition", "emission", "initial_z", "initial_a")
)


@jax.jit
def normalize(vec):
    z = (vec + eps).sum()
    return vec / z, jnp.log(z)


@jax.jit
def backtrace_step(t, transition, actions, alfas, states):
    belief = alfas[t] * transition[actions[t], :, states[t + 1]]
    return states.at[t].set(belief.argmax())


def backtrace(transition, a, alfas):
    states = jnp.zeros_like(a)
    states.at[-1].set(alfas[-1].argmax())
    for t in tqdm(list(range(a.shape[0] - 2, -1, -1))):
        states = backtrace_step(t, transition, a, alfas, states)
    return states


def viterbi_step(hmm_params: HMMParams, observations, actions):
    loglik, alfas = forward(hmm_params, observations, actions)

    states = backtrace(hmm_params.transition, actions, alfas)

    new_transition = np.zeros_like(hmm_params.transition)
    for s1, s2, a in tqdm(list(zip(states[:-1], states[1:], actions[:-1]))):
        # s1->s2 | a
        new_transition[a, s1, s2] += 1

    # reduce
    v = new_transition.sum(axis=0).sum(axis=1).nonzero()[0]
    new_transition = new_transition[:, v, :][:, :, v]
    E = hmm_params.emission[v, :]
    pi_z = hmm_params.initial_z[v]

    print(new_transition.shape)

    # ent = ss.entropy(
    #     hmm_params.transition.ravel(), new_transition.ravel()
    # )

    hmm_params = HMMParams(
        transition=jnp.array(new_transition),
        emission=E,
        initial_a=hmm_params.initial_a,
        initial_z=pi_z,
    )
    return hmm_params, 0


def learn_viterbi_T_hmm(hmm, observations, actions):
    """
    HMM adaptation of the viterbi algorithm
    """
    ents = []
    hmm_params = hmm.hmm_params
    for i in range(100):
        hmm_params, entropy = viterbi_step(hmm_params, observations, actions)
        ents.append(entropy)

        if i % 10 == 0:
            hmm.set_transition(hmm_params.transition)
            hmm.set_emission(hmm_params.emission)
            hmm.set_pi_z(hmm_params.initial_z)
            hmm.save(f"viterbi_{i}")

    return hmm_params


@jax.jit
def _forward_step(emission, transition, observations, actions, alfas, t):
    o_now = observations[t]
    a_prev = actions[t - 1]
    phi_t = emission[:, o_now]
    alfa = phi_t * transition[a_prev].T.dot(alfas[t - 1])
    alfa, log_z = normalize(alfa)
    return alfas.at[t].set(alfa), log_z


def forward(hmm_params, observations, actions=None):
    """
    The forward algorithm for HMM's does filtering:
        alfa[t] propto p(z_t = j | x_{1:t}, a_{1:t})

    based on Murphy's Machine Learning: a probabilistic
        perspective (p612)
    """
    emission = hmm_params.emission
    transition = hmm_params.transition
    pi_z = hmm_params.initial_z

    if actions is None:
        actions = jnp.zeros(len(observations), dtype=jnp.int16)

    alfas = jnp.zeros((len(observations), emission.shape[0]))

    phi_0 = emission[:, observations[0]]
    alfa = phi_0 * pi_z
    alfa, log_lik = normalize(alfa)
    alfas = alfas.at[0].set(alfa)

    log_lik = 0
    for t in tqdm(range(1, len(observations))):
        alfas, log_z = _forward_step(
            emission, transition, observations, actions, alfas, t
        )
        log_lik += log_z

    return log_lik, alfas


@jax.jit
def _backward_step(emission, transition, observations, actions, betas, tn):
    t = tn - 1
    o_tn = observations[tn]
    phi_tn = emission[:, o_tn]
    # Action a_t brought you from state z_t to z_t+1
    # hence it's this we pull in reverse
    a_t = actions[t]
    beta, log_z = normalize(transition[a_t].dot(phi_tn * betas[tn]))
    return betas.at[t].set(beta), log_z


def backward(hmm_params, observations, actions=None):
    """
    The backward algorithm for HMM
        beta[t] = p(x_{t:T}|z_T=i)

    as described in ML: a probabilistic perspective p613
    """
    emission = hmm_params.emission
    transition = hmm_params.transition

    if actions is None:
        actions = jnp.zeros(len(observations), dtype=np.int16)

    n_states = emission.shape[0]
    betas = jnp.zeros((len(observations), n_states))

    beta = jnp.ones(n_states)
    betas = betas.at[len(observations) - 1].set(beta)

    log_lik = 0
    for tn in tqdm(range(1, len(observations))[::-1]):
        betas, log_z = _backward_step(
            emission, transition, observations, actions, betas, tn
        )
        log_lik += log_z

    return log_lik, betas


def forward_backward(hmm_params, observations, actions=None):
    """
    The forward backward algorithm, computing the smoothed posterior marginal
    """
    # run it once to get the jit version
    loglik, alfas = forward(hmm_params, observations, actions)
    _, betas = backward(hmm_params, observations, actions)

    gammas = alfas * betas + eps
    gammas /= gammas.sum(axis=1, keepdims=True)
    return alfas, betas, gammas, loglik


@jax.jit
def _xis_step(emission, transition, observations, actions, alfas, betas, t):
    numerator = (
        alfas[t, :].reshape((*alfas.shape[1:], 1))
        * transition[actions[t]]
        * (emission[:, observations[t + 1]] * betas[t + 1, :]).reshape(
            (1, emission.shape[0])
        )
    )
    denominator = jnp.dot(
        jnp.dot(alfas[t, :].T, transition[actions[t]])
        * emission[:, observations[t + 1]].T,
        betas[t + 1],
    )

    return numerator / (denominator + eps)


def get_xi(hmm_params, observations, actions=None):
    emission = hmm_params.emission
    transition = hmm_params.transition
    pi_a = hmm_params.initial_a
    n_states = emission.shape[0]

    if actions is None:
        actions = np.zeros(len(observations), dtype=np.int16)

    alfas, betas, gammas, loglik = forward_backward(
        hmm_params, observations, actions
    )

    xis = np.zeros((len(pi_a), n_states, n_states)) + eps
    for t in tqdm(range(len(observations) - 1)):
        # the assignment in this array is slow in jax, so use numpy here
        xis[actions[t + 1]] += _xis_step(
            emission, transition, observations, actions, alfas, betas, t
        )

    return alfas, betas, gammas, xis, loglik


class HMM:
    def __init__(self, n_observations, n_states, n_actions, store_path):
        self.n_states = n_states

        self.pi_o = np.ones((n_observations)) / n_observations
        self.pi_a = np.ones((n_actions)) / n_actions
        self.pi_z = np.ones((n_states)) / n_states

        # action, from, to
        self.transition = 1 + np.random.random((n_actions, n_states, n_states))
        self.transition /= self.transition.sum(axis=2, keepdims=True)

        self.emission = 1 + np.random.random((n_states, n_observations))
        self.emission /= self.emission.sum(axis=1, keepdims=True)

        print(self.transition.sum(), self.emission.sum())

        self._store_path = store_path

    @property
    def hmm_params(self):
        return HMMParams(
            transition=self.transition,
            emission=self.emission,
            initial_z=self.pi_z,
            initial_a=self.pi_a,
        )

    def set_pi_o(self, pi_o):
        self.pi_o = pi_o

    def set_pi_a(self, pi_a):
        self.pi_a = pi_a

    def set_pi_z(self, pi_z):
        self.pi_z = pi_z

    def set_transition(self, transition):
        self.transition = transition

    def set_emission(self, emission):
        self.emission = emission

    def _em_step(self, observations, actions):
        ########
        # E Step
        ########
        _, _, gammas, xis, loglik = get_xi(
            self.hmm_params, observations, actions
        )

        ########
        # M Step
        ########
        transition = eps + xis / jnp.sum(gammas[:-1], axis=0).reshape(
            (1, -1, 1)
        )
        # normalize
        transition /= transition.sum(axis=2, keepdims=True)
        self.set_transition(transition)
        logger.info("Transition computed")

        denominator = jnp.sum(gammas, axis=0) + eps
        emission = jnp.zeros_like(self.emission)
        for o_k in range(self.emission.shape[1]):
            emission = emission.at[:, o_k].set(
                np.sum(gammas[observations == o_k, :], axis=0)
            )

        emission = jnp.divide(emission, denominator.reshape((-1, 1))) + eps
        emission /= emission.sum(axis=1, keepdims=True)
        self.set_emission(emission)
        logger.info("Emission computed")
        return loglik

    def save(self, i):
        with open(self._store_path / f"hmm_step_{i}.pkl", "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as outp:
            return pickle.load(outp)

    def fit_baum_welch(
        self,
        observations,
        actions=None,
        n_iters=20,
        save_every=None,
        start_from=None,
    ):
        """
        Expectation Maximimization (EM) for HMM's
        based on p15 of https://web.stanford.edu/~jurafsky/slp3/A.pdf
        """
        if actions is None:
            actions = np.zeros(len(observations), dtype=np.int16)

        if start_from is not None:
            self = self.load(self._store_path / f"hmm_step_{start_from}.pkl")
            start_from = start_from + 1
        else:
            start_from = 0

        self.prev_params = copy.deepcopy(self.hmm_params)
        for i in range(start_from, n_iters):
            print(f"Running step {i:03d}/{n_iters:03d}...")
            loglik = self._em_step(observations, actions)
            print(f"Log likelihood: {loglik}")
            print(self.transition.sum(), self.emission.sum())

            if save_every and i % save_every == 0:
                self.save(i)

            tp_entropy = ss.entropy(
                self.hmm_params.transition.ravel(),
                self.prev_params.transition.ravel(),
            )
            ep_entropy = ss.entropy(
                self.hmm_params.emission.ravel(),
                self.prev_params.emission.ravel(),
            )
            print(f"TP Entropy {tp_entropy}")
            print(f"EP Entropy {ep_entropy}")
            if tp_entropy < 0.001 and ep_entropy < 0.001:
                print("Converged...")
                self.save(i)
                break

            self.prev_params = copy.deepcopy(self.hmm_params)
