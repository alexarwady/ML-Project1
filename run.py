# -*- coding: utf-8 -*-
""" Script file to run to obtain an exact submission
Authors: Victor Faramond, Dario Anongba Varela, Mathieu Schopfer
"""

# Useful starting lines
import numpy as np
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from implementations import ridge_regression
from helpers import get_jet_masks, build_poly, process_data, add_constant_column

print('Script running... Please wait \n')

DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

# Load the training data into feature matrix, class labels, and event ids:
y, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)

# Load the test data into feature matrix, class labels, and event ids:
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Split data in subsets corresponding to a jet value
msks_jet_train = get_jet_masks(tX_train)
msks_jet_test = get_jet_masks(tX_test)

# Ridge regression parameters for each subset
lambdas = [0.002, 0.001, 0.001]
# Polynomial features degree for each subset
degrees = [4, 7, 9]

# Vector to store the final prediction
y_pred = np.zeros(tX_test.shape[0])

for idx in range(len(msks_jet_train)):
    x_train = tX_train[msks_jet_train[idx]]
    x_test = tX_test[msks_jet_test[idx]]
    y_train = y[msks_jet_train[idx]]

    # Pre-processing of data
    x_train, x_test = process_data(x_train, x_test, False)

    phi_train = build_poly(x_train, degrees[idx])
    phi_test = build_poly(x_test, degrees[idx])

    phi_train = add_constant_column(phi_train)
    phi_test = add_constant_column(phi_test)

    weights, loss = ridge_regression(y_train, phi_train, lambdas[idx])

    y_test_pred = predict_labels(weights, phi_test)

    y_pred[msks_jet_test[idx]] = y_test_pred

# We give the name of the output file
OUTPUT_PATH = 'data/output_ridge_regression.csv'

# Generate predictions and save ouput in csv format for submission:
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print('Done !')
