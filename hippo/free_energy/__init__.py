import numpy as np
from pymdp.maths import (
    softmax_obj_arr,
    spm_MDP_G,
    spm_log_single,
    compute_accuracy,
    spm_dot,
)
from pymdp.utils import to_obj_array


def free_energy(qs, prior, likelihood=None):

    free_energy = 0
    # Neg-entropy of posterior marginal H(q[f])
    negH_qs = qs.dot(np.log(qs[:, np.newaxis] + 1e-16))
    # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
    xH_qp = -qs.dot(prior[:, np.newaxis])
    free_energy += negH_qs + xH_qp

    if likelihood is not None:
        free_energy -= compute_accuracy(likelihood, qs)
    return free_energy


def expected_free_energy(agent, last_node, action, constraint, use_epistemic):
    step = last_node.data["step"] + 1

    qs = last_node.data["qs"]
    qs_tau = agent.transition(qs, action)

    # Constraint is set in state space
    C_prob = softmax_obj_arr(np.array([constraint]))
    lnC = spm_log_single(C_prob[0])

    if lnC.shape == qs_tau.shape:
        # state space utility
        expected_utility = qs_tau.dot(lnC)
    else:
        # observation space utility
        qo_tau = spm_dot(agent.likelihood, qs_tau)
        expected_utility = qo_tau.dot(lnC)

    A = to_obj_array(np.array([agent.likelihood]))
    x = to_obj_array(np.array([qs_tau]))
    infogain = spm_MDP_G(A, x)

    nefe = expected_utility + float(int(use_epistemic)) * infogain

    data = {
        "nefe": last_node.data["nefe"] + nefe,
        "expected_utility": expected_utility,
        "infogain": infogain,
        "action": action,
        "qs": qs_tau,
        "state_entropy": -np.sum(qs_tau * np.log(qs_tau)),
        "step": step,
    }

    return nefe, data
