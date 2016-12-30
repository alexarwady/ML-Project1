# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""

import numpy as np


def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(-np.logaddexp(0, -t))


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss_neg_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    s = tx.dot(w)
    l = np.log(1 + tx.dot(w))

    for i in range(len(y)):
        if np.isinf(l[i]):
            l[i] = s[i]

    for i in range(len(y)):
        if l[i] == 0:
            l[i] = np.log(2) + s[i] / 2

    sum = 0

    for i in range(len(y)):
        sum += l[i] - y[i]*(tx[i].dot(w))

    return sum


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
