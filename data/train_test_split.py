# -*- coding: utf-8 -*-
""" This module provides a function for splitting datasets."""

from skmultilearn.model_selection import IterativeStratification

def iterative_train_test(X, y, test_size):
    """
    Iteratively splits data with stratification.

    This function is based on the iterative_train_test_split function from the
    skmultilearn.model_selection package, but uses pandas dataframes as input and output.

    Parameters
    ----------
    X : pandas dataframe
        Data samples.
    y : array or sparse matrix
        Indicator matrix.
    test_size : float [0,1]
        The proportion of the dataset to include in the test split, the rest will be put in the train set.

    Returns
    -------
    X_train : pandas dataframe
        Training samples.
    y_train : array or sparse matrix
        Indicator matrix of the training samples.
    X_test : pandas dataframe
        Test samples.
    y_test : array or sparse matrix
        Indicator matrix of the test samples.

    """
    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X.iloc[train_indexes], y[train_indexes]
    X_test, y_test = X.iloc[test_indexes], y[test_indexes]

    return X_train, y_train, X_test, y_test
