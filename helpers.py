# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def compute_gradient(y, tx, w):
    """ Linear regression using gradient descent. """
    e = y - tx.dot(w)
    n = len(y)

    return -np.dot(tx.T, e) / n


def sigmoid(t):
    """ Apply sigmoid function on t. """
    return np.exp(-np.logaddexp(0, -t))


def standardize(x, mean_x=None, std_x=None):
    """ Standardize the original data set. """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """ Apply a polynomial basis to all the X features. """
    # First, we find the combinations of columns for which we have to
    # compute the product
    m, n = x.shape

    combinations = {}

    # Add combinations of same column power
    for i in range(n * degree):
        if i < n:
            combinations[i] = [i]
        else:
            col_number = i - n
            cpt = 2
            while col_number >= n:
                col_number -= n
                cpt += 1
            combinations[i] = [col_number] * cpt

    # Add combinations of products between columns
    cpt = i + 1

    for i in range(n):
        for j in range(i + 1, n):
            combinations[cpt] = [i, j]
            cpt = cpt + 1

    # Now we can fill a new matrix with the column combinations
    eval_poly = np.zeros(
        shape=(m, n + len(combinations))
    )

    for i, c in combinations.items():
        eval_poly[:, i] = x[:, c].prod(1)

    # Add square root
    for i in range(0, n):
        eval_poly[:, len(combinations) + i] = np.abs(x[:, i]) ** 0.5

    return eval_poly


def add_constant_column(x):
    """ Prepend a column of 1 to the matrix. """
    return np.hstack((np.ones((x.shape[0], 1)), x))


def na(x):
    """ Identifies missing values. """
    return np.any(x == -999)


def impute_data(x_train, x_test):
    """ Replace missing values (NA) by the most frequent value of the column. """
    for i in range(x_train.shape[1]):
        # If NA values in column
        if na(x_train[:, i]):
            msk_train = (x_train[:, i] != -999.)
            msk_test = (x_test[:, i] != -999.)
            # Replace NA values with most frequent value
            values, counts = np.unique(x_train[msk_train, i], return_counts=True)
            # If there are values different from NA
            if (len(values) > 1):
                x_train[~msk_train, i] = values[np.argmax(counts)]
                x_test[~msk_test, i] = values[np.argmax(counts)]
            else:
                x_train[~msk_train, i] = 0
                x_test[~msk_test, i] = 0

    return x_train, x_test


def process_data(x_train, x_test, add_constant_col=True):
    """
    Impute missing data and compute inverse log values of positive columns
    """
    # Impute missing data
    x_train, x_test = impute_data(x_train, x_test)

    inv_log_cols = [0, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]

    # Create inverse log values of features which are positive in value.
    x_train_inv_log_cols = np.log(1 / (1 + x_train[:, inv_log_cols]))
    x_train = np.hstack((x_train, x_train_inv_log_cols))

    x_test_inv_log_cols = np.log(1 / (1 + x_test[:, inv_log_cols]))
    x_test = np.hstack((x_test, x_test_inv_log_cols))

    x_train, mean_x_train, std_x_train = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test, mean_x_train, std_x_train)

    if add_constant_col is True:
        x_train = add_constant_column(x_train)
        x_test = add_constant_column(x_test)

    return x_train, x_test


def get_jet_masks(x):
    """
    Returns 4 masks corresponding to the rows of x with a jet value
    of 0, 1, 2 and 3 respectively.
    """
    return {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)
    }
