## Machine Learning - Project 1

In this repository, you can find my work for the Project 1 of the [Machine Learning](http://mlo.epfl.ch/page-136795.html) course at [EPFL](http://epfl.ch).

This project consists in a [Kaggle competition](https://inclass.kaggle.com/c/epfml-project-1) similar to the [Higgs Boson Machine Learning Challenge (2014)](https://www.kaggle.com/c/Higgs-boson). My team ended at the 3rd place (over 117 teams).

This file explains the organisation and functions of the python scripts. For more information about the implementation, see the [PDF report](https://github.com/vfaramond/ML-Project1/blob/master/Report.pdf) and the commented code.

First, you should place `train.csv` and `test.csv` in a `data` folder at the root of the project.

### `costs.py`
Contain 3 different cost functions like:
- **`calculate_mse`**: Mean square error
- **`calculate_mae`**: Mean absolute error
- **`compute_loss_neg_log_likelihood`**: Negative log likelihood

### `cross_validation.py`
Contain helper methods for cross validation.
- **`build_k_indices`**: Builds k indices for k-fold cross validation
- **`cross_validation_visualization`**: Creates a plot showing the accuracy given a lambda value

### `helpers.py`
Contain multiple methods for data processing and utilitary methods necessary to achieve the regression methods:
- **`standardize`, `buid_poly`, `add_constant_column`, `na`, `impute_data` and `process_data`**: All the processing functions. See the report for explications about those functions.
- **`compute_gradient`**: Computes the gradient for gradient descent and stochastic gradient descent
- **`batch_iter`**: Generate a minibatch iterator for a dataset

### `proj1_helpers.py`
Contain functions used to load the data and generate a CSV submission file.

### `implementations.py`
Contain the 6 regression methods needed for this project
- **`least_squares_gd`**: Linear regression using gradient descent
- **`least_squares_sgd`**: Linear regression using stochastic gradient descent
- **`least_squares`**: Least squares regression using normal equations
- **`ridge_regression`**: Ridge regression using normal equations
- **`logistic_regression`**: using stochastic gradient descent
- **`reg_logistic_regression`**: Regularized logistic regression

### `run.py`
Script that generates the exact CSV file submitted on Kaggle.

### `project1.ipynb`
Python notebook used for tests during this project.
