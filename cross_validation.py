# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import predict_labels, compute_accuracy
from helpers import process_data, get_jet_masks, build_poly, add_constant_column
from implementations import ridge_regression


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, regression_method, **args):
    """
    Completes k-fold cross-validation using the regression method
    passed as argument.
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train = x[msk_train, :]
    x_test = x[msk_test, :]
    y_train = y[msk_train]
    y_test = y[msk_test]

    # data pre-processing
    x_train, x_test = process_data(x_train, x_test, True)

    # compute weights using given method
    weights, loss = regression_method(y=y_train, tx=x_train, **args)

    # predict output for train and test data
    y_train_pred = predict_labels(weights, x_train)
    y_test_pred = predict_labels(weights, x_test)

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train)
    acc_test = compute_accuracy(y_test_pred, y_test)

    return acc_train, acc_test


def cross_validation_ridge_regression(y, x, k_indices, k, lambdas, degrees):
    """
    Completes k-fold cross-validation using the ridge regression method.
    Here, we build polynomial features and create four subsets using
    the jet feature.
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train_all_jets = x[msk_train, :]
    x_test_all_jets = x[msk_test, :]
    y_train_all_jets = y[msk_train]
    y_test_all_jets = y[msk_test]

    # split in 4 subsets the training set
    msk_jets_train = get_jet_masks(x_train_all_jets)
    msk_jets_test = get_jet_masks(x_test_all_jets)

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train_all_jets))
    y_test_pred = np.zeros(len(y_test_all_jets))

    for idx in range(len(msk_jets_train)):
        x_train = x_train_all_jets[msk_jets_train[idx]]
        x_test = x_test_all_jets[msk_jets_test[idx]]
        y_train = y_train_all_jets[msk_jets_train[idx]]

        # data pre-processing
        x_train, x_test = process_data(x_train, x_test, False)

        phi_train = build_poly(x_train, degrees[idx])
        phi_test = build_poly(x_test, degrees[idx])

        phi_train = add_constant_column(phi_train)
        phi_test = add_constant_column(phi_test)

        # compute weights using given method
        weights, loss = ridge_regression(y=y_train, tx=phi_train, lambda_=lambdas[idx])

        y_train_pred[msk_jets_train[idx]] = predict_labels(weights, phi_train)
        y_test_pred[msk_jets_test[idx]] = predict_labels(weights, phi_test)

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train_all_jets)
    acc_test = compute_accuracy(y_test_pred, y_test_all_jets)

    return acc_train, acc_test


def cross_validation_visualization(lambds, acc_train, acc_test):
    """visualization the curves of acc_train and acc_test."""
    plt.semilogx(lambds, acc_train, marker=".", color='b', label='train error')
    plt.semilogx(lambds, acc_test, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=10)
    plt.grid(True)
