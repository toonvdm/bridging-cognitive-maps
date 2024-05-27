import numpy as np


def add_death(mat, eps=1e-15):
    """
    Given an "illigal" non-stay action, there is eps chance to die
    In the paper this is referred to as a dummy state.
    """
    for action in range(mat.shape[-1]):
        zer = np.where(mat[..., action].sum(axis=0) == 0)[0]
        mat[-1, zer, action] = eps
    # Stay dead once reached
    mat[-1, -1, :] = 1.0
    return mat


def add_stationary(mat, eps=1e-15):
    """
    Add an action to stand still
    """
    return np.concatenate(
        [mat, np.eye(mat.shape[0]).reshape(*mat[..., :1].shape)], axis=2
    )


def extract_AB(chmm, reduce=False, do_add_stationary=False, do_add_death=True):
    death = int(do_add_death)
    n_obs = len(chmm.n_clones) + death

    T = chmm.T.transpose(2, 1, 0)
    unreduced_n_states = T.shape[0]

    if reduce:
        v = T.sum(axis=2).sum(axis=1).nonzero()[0]
        T = T[v, :][:, v]

    # Transition matrix
    B = np.zeros((T.shape[0] + death, T.shape[0] + death, T.shape[2]))
    B[: T.shape[0], : T.shape[1]] = T
    if do_add_death:
        B = add_death(B)
    if do_add_stationary:
        B = add_stationary(B)

    # Normalize this dude
    B /= B.sum(axis=0, keepdims=True)

    # A matrix = likelihood matrix, unreduced matrix and uniform probabilities
    state_loc = np.hstack(
        (np.array([0], dtype=chmm.n_clones.dtype), chmm.n_clones)
    ).cumsum()

    A = np.zeros((n_obs, unreduced_n_states + death))
    for i in range(n_obs - int(death)):
        s, f = state_loc[i : i + 2]
        A[i, s:f] = 1.0

    # Direct mapping of state to a death observation
    if do_add_death:
        A[-1, -1] = 1.0

    if reduce and do_add_death:
        v = np.concatenate([v, np.array([-1])])

    # Only consider the reduced states
    if reduce:
        A = A[:, v]

    # Normalize over state: Sum_s P(o|s) = 1
    A /= A.sum(axis=0, keepdims=True)

    return A, B
