# -*- coding: utf-8 -*-
""" Code for running the SVM models. """

import numpy as np
from data.data_helper import load_instances, load_bert_instances, load_subsets
from embedding.glove_sentence_embeddings import GloVeEmbeddingTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_fscore_support
from metrics.hierarchical_performance_measures import hierarchical_eval, get_performance
from hierarchicalSVM import hierSVM, get_predictions
from scipy.special import expit

if __name__ == "__main__":

    # Data parameters
    holdout = False
    bert = False
    pooling = 'max'

    # Classifier parameters
    c_flat = 1e-3
    thresh_flat = 0.8
    c_hier = 1e-3
    thresh_hier = [0, 0, 0.25]
    class_weight = 'balanced'
    loss = 'hinge'
    tol = 1e-1
    max_iter = 10000

    if bert:
        X_train, X_test = load_bert_instances(holdout, pooling)
    else:
        # convert text to Glove embeddings
        train, test = load_instances(holdout)
        glove_emb = GloVeEmbeddingTransformer()
        glove_emb.fit(train)
        X_train = glove_emb.transform(train)
        X_test = glove_emb.transform(test)

    (y_train_lvl1, y_test_lvl1, y_train_lvl2, y_test_lvl2, y_train_lvl3,
     y_test_lvl3, encoder_lvl1, encoder_lvl2, encoder_lvl3,
     number_of_classes_L1, number_of_classes_L2, number_of_classes_L3,
     L2_train_ind, L2_test_ind, L2_y_labels, L2_y_col_ind,
     L3_train_ind, L3_test_ind, L3_y_labels, L3_y_col_ind) = load_subsets(holdout)

    # Scale data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train flat SVM classifier
    flat_clf = OneVsRestClassifier(LinearSVC(C=c_flat, class_weight=class_weight, max_iter=max_iter, tol=tol, loss=loss))
    flat_clf.fit(X_train_sc, y_train_lvl3)

    y_dec_func = flat_clf.decision_function(X_test_sc)
    y_bin_pred = csr_matrix(np.where(expit(np.array(y_dec_func)) > thresh_flat, 1, 0))

    # Evaluate performance
    macroP, macroR, macroF1, _ = precision_recall_fscore_support(y_test_lvl3, y_bin_pred, average='macro')
    hP, hR, hF1 = hierarchical_eval(y_test_lvl3, y_bin_pred, encoder_lvl3)

    # Train hierarchical SVM classifier
    hier_clf1, hier_clf2, hier_clf3 = hierSVM(X_train_sc, y_train_lvl1, y_train_lvl2, y_train_lvl3,
                                              number_of_classes_L1, number_of_classes_L2, number_of_classes_L3,
                                              L2_train_ind, L2_y_col_ind, L3_train_ind, L3_y_col_ind,
                                              c_hier, class_weight='balanced', max_iter=max_iter, tol=tol, loss=loss)

    y_pred_lvl1, y_pred_lvl2, y_pred_lvl3 = get_predictions(X_test_sc, hier_clf1, hier_clf2, hier_clf3, number_of_classes_L3, thresh_hier)

    # Evaluate performance
    hier_performance = get_performance(y_train_lvl1, y_train_lvl2, y_train_lvl3,
                                       y_pred_lvl1, y_pred_lvl2, y_pred_lvl3,
                                       encoder_lvl1, encoder_lvl2, encoder_lvl3)
