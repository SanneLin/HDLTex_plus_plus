# -*- coding: utf-8 -*-
""" This module provides a function for calculating class weights. """

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculating_class_weights(y_true):
    """
    Calculates weights given to positive and negative examples of each class.

    Parameters
    ----------
    y_true : NumPy array
        label (indicator) matrix.

    Returns
    -------
    weights : NumPy array
        array of weights.

    """
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', classes=[0.,1.], y=y_true[:, i].toarray().ravel())
    return weights