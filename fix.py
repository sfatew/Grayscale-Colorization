import numpy as np
# import torch


def compute_rebalance_weights(p, Q=313, lam=0.5):
    """
    p: numpy array of shape (313,), empirical distribution over color bins
    returns: weight array of shape (313,)
    """
    mixed = (1 - lam) * p + lam / Q
    w = 1 / mixed
    w *= Q / np.sum(w * p)  # normalize to expected value 1
    return w

prior_probs = np.load('util/prior_probs.npy')

pts_in_hull = np.load('util/pts_in_hull.npy')

rebalance_weights = compute_rebalance_weights(prior_probs)  # shape: (313,)
# rebalance_weights = torch.tensor(rebalance_weights, dtype=torch.float32)
inverse_weights = compute_rebalance_weights(prior_probs, lam=0)  # shape: (313,)


# print(pts_in_hull)
print(inverse_weights)


