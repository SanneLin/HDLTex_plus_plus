# -*- coding: utf-8 -*-
""" This module contains the function for calculating the weighted loss function. """

from tensorflow.python.keras import backend as K

def get_weighted_loss(weights):
    """
    Calculates the weighted binary cross-entropy loss function.

    Parameters
    ----------
    weights : NumPy array
        array of weights.

    Returns
    -------
    float
        weighted loss.

    """
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss