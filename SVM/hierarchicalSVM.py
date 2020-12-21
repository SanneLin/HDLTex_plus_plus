# -*- coding: utf-8 -*-
""" This module provides function for training and evaluating the hierarchical SVM classifier. """

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from scipy.special import expit

def hierSVM(X, y_train_lvl1, y_train_lvl2, y_train_lvl3,
            number_of_classes_L1, number_of_classes_L2, number_of_classes_L3,
            L2_train_ind, L2_y_col_ind, L3_train_ind, L3_y_col_ind,
            C, class_weight='balanced', **kwargs):
    """
    Fits each submodel of the hierarchical SVM classifier.

    Parameters
    ----------
    X : array-like
        Data matrix.
    y_train_lvl1, y_train_lvl2, y_train_lvl3 : sparse matrix
        Primary, secondary, and tertiary level indicator matrices for the train (or train+validation) dataset.
    number_of_classes_L1 : int
        Number of primary categories.
    number_of_classes_L2 : list of int
        Number of secondary categories per primary category.
    number_of_classes_L3 : list of int
        Number of tertiary categories per secondary category.
    L2_train_ind : list of list of int
        Contains for each primary category the list of row indices of training instances belonging
        to the category's descendant categories.
    L2_y_col_ind : list of list of int
        Contains for each primary category the list of column indices of y_train_lvl2 which
        correspond to the category's child categories.
    L3_train_ind : list of list of int
        Contains for each secondary category the list of row indices of training instances belonging
        to the category's child categories.
    L3_y_col_ind : list of list of int
        Contains for each secondary category the list of column indices of y_train_lvl3 which
        correspond to the category's child categories.
    C : float
        Regularization parameter.
    class_weight : dict or ‘balanced’, optional
        Set the parameter C of class i to class_weight[i]*C for SVC. If not given,
        all classes are supposed to have weight one. The “balanced” mode uses the
        values of y to automatically adjust weights inversely proportional to class
        frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        The default is 'balanced'.
    **kwargs
        Other functions arguments to be passed to the SVM classifier.

    Returns
    -------
    clf_L1 : OneVsRestClassifier object
        Primary level submodel.
    clf_L2 : list of OneVsRestClassifier object
        Secondary level submodels.
    clf_L3 : list of OneVsRestClassifier object
        Tertiary level submodels.

    """

    clf_L1 = OneVsRestClassifier(LinearSVC(C=C, class_weight=class_weight, **kwargs))
    clf_L1.fit(X, y_train_lvl1)

    clf_L2 = [None] * number_of_classes_L1
    for i in range(number_of_classes_L1):
        clf_L2[i] = OneVsRestClassifier(LinearSVC(C=C, class_weight=class_weight, random_state=1, **kwargs))
        clf_L2[i].fit(X[L2_train_ind[i]], y_train_lvl2[L2_train_ind[i]][:,L2_y_col_ind[i]])

    clf_L3 = [None] * len(number_of_classes_L3)
    for i in range(len(number_of_classes_L3)):
        if len(L3_y_col_ind[i]) == 1:
            # if a L2 node only has one child node, SVM cannot be trained on the child category
            # (no negative examples using the exclusive siblings subsetting method)
            continue
        clf_L3[i] = OneVsRestClassifier(LinearSVC(C=C, class_weight=class_weight, random_state=1, **kwargs))
        clf_L3[i].fit(X[L3_train_ind[i]], y_train_lvl3[L3_train_ind[i]][:,L3_y_col_ind[i]])

    return clf_L1, clf_L2, clf_L3

def get_decision_matrices(X, clf_L1, clf_L2, clf_L3, number_of_classes_L3):
    """
    Returns the decision matrices at each level of the hierarchical SVM model.

    Parameters
    ----------
    X : array-like
        Data matrix.
    clf_L1 : OneVsRestClassifier object
        Primary level submodel.
    clf_L2 : list of OneVsRestClassifier object
        Secondary level submodels.
    clf_L3 : list of OneVsRestClassifier object
        Tertiary level submodels.
    number_of_classes_L3 : list of int
        Number of tertiary categories per secondary category.

    Returns
    -------
    y_func_L1, y_func_L2, y_func_L3 : array
        Primary, secondary, and tertiary level decision matrices.

    """
    y_func_L1 = clf_L1.decision_function(X)
    y_func_L2 = clf_L2[0].decision_function(X)

    for i in range(len(clf_L2)):
        if i == 0:
            y_func_L2 = clf_L2[i].decision_function(X)
        else:
            y_func_L2 = np.hstack((y_func_L2, clf_L2[i].decision_function(X)))

    for i in range(len(clf_L3)):
        if i == 0:
            y_func_L3 = clf_L3[i].decision_function(X)
        elif len(number_of_classes_L3[i]) == 1:
            # If no classifier for a certain node, set value arbitrarily high, such that
            # prediction depends entirely on the output of the classifier one level higher
            y_func_L3 = np.hstack((y_func_L3, np.array([100]*X.shape[0]).reshape(X.shape[0],1)))
        else:
            y_func_L3 = np.hstack((y_func_L3, clf_L3[i].decision_function(X)))

    return y_func_L1, y_func_L2, y_func_L3

def get_predictions(X, clf_L1, clf_L2, clf_L3, number_of_classes_L3, thresh):
    """
    Returns the predicted indicator matrices at each level of the hierarchical SVM model.

    Parameters
    ----------
    X : array-like
        Data matrix.
    clf_L1 : OneVsRestClassifier object
        Primary level submodel.
    clf_L2 : list of OneVsRestClassifier object
        Secondary level submodels.
    clf_L3 : list of OneVsRestClassifier object
        Tertiary level submodels.
    number_of_classes_L3 : list of int
        Number of tertiary categories per secondary category.
    thresh : list of float
        Threshold values.

    Returns
    -------
    y_pred_bin_L1, y_pred_bin_L2, y_pred_bin_L3 : array
        Primary, secondary, and tertiary level predicted indicator matrices.

    """
    y_func_L1, y_func_L2, y_func_L3 = get_decision_matrices(X, clf_L1, clf_L2, clf_L3, number_of_classes_L3)

    y_pred_bin_L1 = np.where(expit(y_func_L1) > thresh[0], 1, 0)
    y_pred_bin_L2 = np.where(expit(y_func_L2) > thresh[1], 1, 0)
    y_pred_bin_L3 = np.where(expit(y_func_L3) > thresh[2], 1, 0)

    return y_pred_bin_L1, y_pred_bin_L2, y_pred_bin_L3