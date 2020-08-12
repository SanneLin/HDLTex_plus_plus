# -*- coding: utf-8 -*-
""" Performance measures for evaluating predictions with a hierarchical structure. """

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

def hierarchical_eval(y_true, y_pred, fitted_encoder, *, rm_cat=None):
    """
    Computes the hierarchical precision, recall, and F1 for a predictions with a three-level hierarchy.

    Parameters
    ----------
    y_true : sparse matrix
        True indicator matrix.
    y_pred : array
        Predicted indicator matrix.
    fitted_encoder : MultiLabel Binarizer object
        Encoder that can convert indicator matrices into sets of labels and vice versa.
    rm_cat : list of str, optional
        Set of categories to be excluded when computing performance of the tertiary level predictions. The default is None.

    Returns
    -------
    hP, hR, hF1 : float
        Hierarchical precision, recall, and F1.

    """
    y_true = fitted_encoder.inverse_transform(y_true)
    y_pred = fitted_encoder.inverse_transform(y_pred)

    num_intersection = 0
    num_actual = 0
    num_predict = 0

    def append_int_nodes(jel_level3):
        jel_level1 = sorted({i[:1] for i in jel_level3})
        jel_level2 = sorted({i[:2] for i in jel_level3})
        return set(jel_level3).union(jel_level1, jel_level2)

    for i,j in zip(y_true, y_pred):
        if rm_cat:
            set_i = set(i).difference(rm_cat)
            set_j = set(j).difference(rm_cat)
        else:
            set_i = set(i)
            set_j = set(j)

        ext1 = append_int_nodes(set_i)
        ext2 = append_int_nodes(set_j)

        num_intersection += len(set(ext1.intersection(ext2)))
        num_actual += len(set(ext1))
        num_predict += len(set(ext2))

    if num_predict == 0:
        print('hierP undefined - division by zero')
        hP = None
    else:
        hP = num_intersection/num_predict

    if num_actual == 0:
        print('hierR undefined - division by zero')
        hR = None
    else:
        hR = num_intersection/num_actual

    if (hP is None) or (hR is None):
        print('hierF1 undefined due to hierP or hierR undefined')
        hF1 = None
    elif (hP + hR == 0):
        print('hierF1 undefined because hP+hR = 0')
        hF1 = None
    else:
        hF1 = (2*hP*hR)/(hP+hR)

    return hP, hR, hF1

def hierarchical_eval_L2(y_true, y_pred, fitted_encoder):
    """
    Computes the hierarchical precision, recall, and F1 for a predictions with a two-level hierarchy.

    Parameters
    ----------
    y_true : sparse matrix
        True indicator matrix.
    y_pred : array
        Predicted indicator matrix.
    fitted_encoder : MultiLabel Binarizer object
        Encoder that can convert indicator matrices into sets of labels and vice versa.

    Returns
    -------
    hP, hR, hF1 : float
        Hierarchical precision, recall, and F1.

    """
    y_true = fitted_encoder.inverse_transform(y_true)
    y_pred = fitted_encoder.inverse_transform(y_pred)
    num_intersection = 0
    num_actual = 0
    num_predict = 0

    def append_int_nodes(jel_level2):
        jel_level1 = sorted({i[:1] for i in jel_level2})
        return set(jel_level2).union(jel_level1)

    for i,j in zip(y_true, y_pred):
        ext1 = append_int_nodes(set(i))
        ext2 = append_int_nodes(set(j))

        num_intersection += len(set(ext1.intersection(ext2)))
        num_actual += len(set(ext1))
        num_predict += len(set(ext2))

    if num_predict == 0:
        print('hierP undefined - division by zero')
        hP = None
    else:
        hP = num_intersection/num_predict

    if num_actual == 0:
        print('hierR undefined - division by zero')
        hR = None
    else:
        hR = num_intersection/num_actual

    if (hP is None) or (hR is None):
        print('hierF1 undefined due to hierP or hierR undefined')
        hF1 = None
    elif (hP + hR == 0):
        print('hierF1 undefined because hP+hR = 0')
        hF1 = None
    else:
        hF1 = (2*hP*hR)/(hP+hR)

    return hP, hR, hF1

def get_performance(y_true_L1, y_true_L2, y_true_L3, y_pred_bin_L1, y_pred_bin_L2, y_pred_bin_L3, enc_L1, enc_L2, enc_L3):
    """
    Evaluates the precision, recall, and F1 scores of a model's predictions.

    Parameters
    ----------
    y_true_L1, y_true_L2, y_true_L3 : sparse matrix
        True indicator matrices.
    y_pred_bin_L1, y_pred_bin_L2, y_pred_bin_L3 : array
        Predicted indicator matrices.
    enc_L1, enc_L2, enc_L3 : MultiLabel Binarizer object
        Multilabel encoders for the primary, secondary, and tertiary level labels.

    Returns
    -------
    performance : list of dict
        Performance at each level of the hierarchy.

    """
    y_pred_L1 = enc_L1.inverse_transform(y_pred_bin_L1)
    y_pred_L2 = enc_L2.inverse_transform(y_pred_bin_L2)
    y_pred_L3 = enc_L3.inverse_transform(y_pred_bin_L3)

    assert len(y_pred_L1) == len(y_pred_L2) == len(y_pred_L3)

    y_pred_corr_L2 = [None]*len(y_pred_L2)
    for index, (i, j) in enumerate(zip(y_pred_L1, y_pred_L2)):
        corr = [label for label in j if label[0:1] in i]
        y_pred_corr_L2[index] = corr

    y_pred_corr_L3 = [None]*len(y_pred_L3)
    for index, (i, j, k) in enumerate(zip(y_pred_L1, y_pred_L2, y_pred_L3)):
        corr = [label for label in k if label[0:2] in j and label[0:1] in i]
        y_pred_corr_L3[index] = corr

    y_pred_bin_corr_L2 = enc_L2.transform(y_pred_corr_L2)
    y_pred_bin_corr_L3 = enc_L3.transform(y_pred_corr_L3)

    performance = []

    # Performance of L1 classifier
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_L1, y_pred_bin_L1, average='macro')
    hP, hR, hF1, _ = precision_recall_fscore_support(y_true_L1, y_pred_bin_L1, average='micro')
    performance.append({
        'Level':1, 'Corrected':False,
        'MacroP':prec, 'MacroR':rec, 'MacroF1': f1,
        'hierP':hP, 'hierR':hR, 'hierF1':hF1})

    # Performance of L2 classifier
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_L2, y_pred_bin_L2, average='macro')
    hP, hR, hF1 = hierarchical_eval(y_true_L2, y_pred_bin_L2, enc_L2)
    performance.append({
        'Level':2, 'Corrected':False,
        'MacroP':prec, 'MacroR':rec, 'MacroF1': f1,
        'hierP':hP, 'hierR':hR, 'hierF1':hF1})

    # Performance of L1+L2 classifiers combined
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_L2, y_pred_bin_corr_L2, average='macro')
    hP, hR, hF1 = hierarchical_eval(y_true_L2, y_pred_bin_corr_L2, enc_L2)
    performance.append({
        'Level':2, 'Corrected':True,
        'MacroP':prec, 'MacroR':rec, 'MacroF1': f1,
        'hierP':hP, 'hierR':hR, 'hierF1':hF1})

    # Performance of L3 classifier
    prec, rec, f1 = adj_prf1(y_true_L3, y_pred_bin_L3, enc_L3, rm={'B00', 'H00', 'I00', 'K00', 'L00', 'M00', 'R00', 'Z00'})
    hP, hR, hF1 = hierarchical_eval(y_true_L3, y_pred_bin_L3, enc_L3, rm_single=True)
    performance.append({
        'Level':3, 'Corrected':False, 'Small':True,
        'MacroP':prec, 'MacroR':rec, 'MacroF1': f1,
        'hierP':hP, 'hierR':hR, 'hierF1':hF1})

    # Performance of L1+L2+L3 classifiers combined (excluding categories not predicted by L3 classifier)
    prec, rec, f1 = adj_prf1(y_true_L3, y_pred_bin_corr_L3, enc_L3, rm={'B00', 'H00', 'I00', 'K00', 'L00', 'M00', 'R00', 'Z00'})
    hP, hR, hF1 = hierarchical_eval(y_true_L3, y_pred_bin_corr_L3, enc_L3, rm_single=True)
    performance.append({
        'Level':3, 'Corrected':True, 'Small':True,
        'MacroP':prec, 'MacroR':rec, 'MacroF1': f1,
        'hierP':hP, 'hierR':hR, 'hierF1':hF1})

    # Performance of overall model
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_L3, y_pred_bin_corr_L3, average='macro')
    hP, hR, hF1 = hierarchical_eval(y_true_L3, y_pred_bin_corr_L3, enc_L3)
    performance.append({
        'Level':3, 'Corrected':True, 'Small':False,
        'MacroP':prec, 'MacroR':rec, 'MacroF1': f1,
        'hierP':hP, 'hierR':hR, 'hierF1':hF1})

    return performance

def adj_prf1(y_true, y_pred, fitted_encoder, rm={}):
    """
    Calculates the precision, recall, and f1 score of a subset of categories.

    Parameters
    ----------
    y_true : sparse matrix
        True indicator matrix.
    y_pred : array
        Predicted indicator matrix.
    fitted_encoder : MultiLabel Binarizer object
        Encoder that converts between indicator matrix and sets of labels and vice versa.
    rm : set, optional
        Set of categories to leave out of evaluation. The default is {}.

    Returns
    -------
    precision : float
        Macro-averaged precision.
    recall : float
        Macro-averaged recall.
    f1 : float
        Macro-averaged f1.

    """
    y_true = fitted_encoder.inverse_transform(y_true)
    y_pred = fitted_encoder.inverse_transform(y_pred)

    classes = set(fitted_encoder.classes_).difference(rm)
    mlb = MultiLabelBinarizer(classes=list(classes))
    mlb.fit(y_true)
    adj_y_true = mlb.transform(y_true)
    adj_y_pred = mlb.transform(y_pred)

    p, r, f1, _ = precision_recall_fscore_support(adj_y_true, adj_y_pred, average='macro')

    return p, r, f1