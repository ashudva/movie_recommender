"""Mixture model using EM"""
from operator import pos
from typing import AsyncIterable, Tuple
from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import var
from common import GaussianMixture


def gaussian(X: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the probablity of vector X under a Gaussian Distribution
    Function supports both single vector and tiled vector
    Case1: When X is vector
        gaussian() returns a float i.e probability of that vector under normal distribution
    Case2: When X is a (K, d) tiled vector
        gaussian() returns a (K,) array of probabilities of untiled vector X for each mixture component
        under a normal distribution

    Args:
        X: (d, ) array or (K, d) tiled-array holding the vector's coordinates
        mean: (d, ) array or (K, d) array of mean of the gaussian
        var: variance or (K,) array of variance of the gaussian

    Returns:
        prob: float or (K,) array of the probability
    """
    # Check if X is a tiled array
    if X.ndim == 2:
        d = X.shape[1]
        log_prob = -d / 2.0 * np.log(2 * np.pi * var)
        # Sum along axis = 1
        log_prob -= 0.5 * ((X - mean)**2).sum(axis=1) / var
    else:
        d = len(X)
        log_prob = -d / 2.0 * np.log(2 * np.pi * var)
        # there's only one axis
        log_prob -= 0.5 * ((X - mean)**2).sum() / var
    return np.exp(log_prob)


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, _ = X.shape
    K, _ = mixture.mu.shape

    # Posterior probabilities
    post = np.zeros((n, K), dtype=np.float)

    # Log likelihood
    ll = 0
    for i in range(n):
        # tiled vector = K times repeats vector X[i] along axis-0 and 1 times along axis-1
        tiled_vector = np.tile(X[i], (K, 1))
        post[i, :] = mixture.p * \
            gaussian(tiled_vector, mixture.mu, mixture.var)
        likelihood = post[i].sum()
        post[i] /= likelihood
        ll += np.log(likelihood)

    return post, ll




def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset.

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, K = post.shape
    d = X.shape[0]
    # n_hat: (K, 1) array of expected no. of points in each component
    n_hat = post.sum(axis=0)
    p = n_hat / n  # (K, 1) mixture weights
    # Updated mus: [K, n] @ [n, d] -> [K, d]
    # taking the advantage numpy broadcasting
    mu = post.T @ X / n_hat
    var = np.zeros(K)
    for j in range(K):
        sse = ((X - mu[j])**2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    c = 1e-6
    prev_ll = None
    ll = None

    while (prev_ll is None or prev_ll - ll >= c * np.abs(ll)):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll
