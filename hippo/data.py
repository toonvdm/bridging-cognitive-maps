import numpy as np

import torch

from pathlib import Path


def mse(a, b):
    return np.linalg.norm(a.copy().flatten() - b.copy().flatten())


class Tokenizer:
    def __init__(self, observations):
        self.codebook = np.zeros((0, *observations[0].shape), dtype=np.int16)
        for o in observations:
            for idx, c in enumerate(self.codebook):
                if mse(c, o) == 0:
                    break
            else:
                self.codebook = np.concatenate(
                    [self.codebook, o.reshape(1, *o.shape)], axis=0
                )

    def __call__(self, observations, block_size=1000, device="cpu"):
        o = torch.tensor(np.expand_dims(observations, 1), dtype=torch.float16)
        c = torch.tensor(
            np.expand_dims(self.codebook.copy(), 0), dtype=torch.float16
        )

        o, c = o.to(device), c.to(device)

        # more does not fit in memory:
        # so has to be computed in blocks
        with torch.no_grad():
            x = np.zeros(observations.shape[0])
            for bi in range(o.shape[0] // block_size + 1):
                v = o[bi * block_size : (bi + 1) * block_size] - c
                if v.shape[0] > 0:
                    v = v.reshape((v.shape[0], c.shape[1], -1))
                    err = torch.linalg.norm(v, dim=-1)
                    x[bi * block_size : (bi + 1) * block_size] = (
                        err.argmin(axis=-1).detach().cpu().numpy()
                    )
                del v

        del o
        del c

        return x.astype(np.int64)

    def get_observation(self, idx):
        if idx < self.codebook.shape[0]:
            return self.codebook[idx]
        else:
            return np.zeros_like(self.codebook[0])


def get_tokenizer(observations, device="cpu", block_size=1000):
    tokenizer = Tokenizer(observations)
    tokens = tokenizer(observations, block_size, device)
    return tokenizer, tokens


def load_data(f, tokenizer=None, device=None, drop_last=True, block_size=1000):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data = np.load(f)
    o, a = data["obs"], data["action"]
    if not tokenizer:
        tokenizer, x_states = get_tokenizer(
            o, block_size=block_size, device=device
        )
    else:
        x_states = tokenizer(o, block_size=block_size, device=device)

    if drop_last:
        x_states = x_states[:-1]
        o = o[:-1]
    return x_states, a, o, tokenizer


def load_multiple_episodes(dir, tokenizer=None, device=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    states, actions, observations = [], [], []
    for f in Path(dir).glob("*.npz"):
        data = np.load(f)
        o, a = data["obs"], data["action"]
        if not tokenizer:
            tokenizer, x_states = get_tokenizer(o)
        else:
            x_states = tokenizer(o, device=device)

        states.append(x_states)
        actions.append(a)
        observations.append(o)

    states = np.array(states)
    actions = np.array(actions)
    observations = np.array(observations)

    return states, actions, observations, tokenizer
