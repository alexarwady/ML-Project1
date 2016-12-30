# -*- coding: utf-8 -*-
"""
Project 1 method implementations.
Authors: Victor Faramond, Dario Anongba Varela, Mathieu Schopfer
"""

import numpy as np
from costs import compute_loss, compute_loss_neg_log_likelihood
from helpers import compute_gradient, batch_iter, sigmoid


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent
    """
    # if initial_w is None, we initialize it to a zeros vector
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    # Define parameters to store weight and loss
    loss = 0
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w -= gamma * gradient

    return w, loss


def least_squares_sgd(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent
    """
    # if initial_w is None, we initialize it to a zeros vector
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    # Define parameters of the algorithm
    batch_size = 1

    # Define parameters to store w and loss
    loss = 0
    w = initial_w

    for n_iter, [mb_y, mb_tx] in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        # compute gradient and loss
        gradient = compute_gradient(mb_y, mb_tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w -= gamma * gradient

    return w, loss


def least_squares(y, tx):
    """ Least squares regression using normal equations
    """
    x_t = tx.T

    w = np.dot(np.dot(np.linalg.inv(np.dot(x_t, tx)), x_t), y)
    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)

    w = np.dot(np.dot(np.linalg.inv(np.dot(x_t, tx) + lambd * np.eye(tx.shape[1])), x_t), y)
    loss = compute_loss(y, tx, w)

    return w, loss


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss_neg_log_likelihood(y, tx, w)
    gradient = np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)

    w -= gamma * gradient

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    y = (1 + y) / 2
    losses = []
    threshold = 0.1

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    y = (1 + y) / 2
    losses = []
    threshold = 0.1

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    norm = sum(w ** 2)
    cost = w + lambda_ * norm / (2 * np.shape(w)[0])

    return w, cost
