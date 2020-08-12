# -*- coding: utf-8 -*-
"""
HDLTex++ model.

Extended from https://github.com/kk7nc/HDLTex/blob/master/HDLTex/HDLTex.py.
"""

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.data_helper import load_subsets, load_data_and_tokenize, load_glove
from build_model import buildModel_RNN, buildModel_CNN
from calculate_weights import calculating_class_weights
from custom_loss import get_weighted_loss
from metrics.hierarchical_performance_measures import get_performance

if __name__ == "__main__":

    # Set parameters
    MAX_SEQUENCE_LENGTH = 500
    BATCH_SIZE = 128
    epochs = 1

    L1_model = 1 # CNN if L1_model is 1, RNN otherwise
    L2_model = 1 # CNN if L2_model is 1, RNN otherwise
    L3_model = 1 # CNN if L3_model is 1, RNN otherwise

    thresh_L1 = 0.6
    thresh_L2 = 0.6
    thresh_L3 = 0.6

    np.set_printoptions(threshold=np.inf)

    # Load data
    holdout = False

    (y_train_lvl1, y_train_lvl2, y_train_lvl3, y_test_lvl1, y_test_lvl2, y_test_lvl3,
     encoder_lvl1, encoder_lvl2, encoder_lvl3, number_of_classes_L1, number_of_classes_L2,
     number_of_classes_L3, L2_train_ind, L2_test_ind, L2_y_labels, L2_y_col_ind,
     L3_train_ind, L3_test_ind, L3_y_labels, L3_y_col_ind) = load_subsets(holdout)

    sequences_train, sequences_test, word_index = load_data_and_tokenize(holdout)
    embeddings_index = load_glove()

    X_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    X_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')

    # Train level 1 model
    if L1_model == 1:
        print('Create CNN model')
        HDLTex_lvl1 = buildModel_CNN(word_index, embeddings_index, number_of_classes_L1, MAX_SEQUENCE_LENGTH)
    else:
        print('Create RNN model')
        HDLTex_lvl1 = buildModel_RNN(word_index, embeddings_index, number_of_classes_L1, MAX_SEQUENCE_LENGTH)
    class_weights = calculating_class_weights(y_train_lvl1)
    HDLTex_lvl1.compile(loss=get_weighted_loss(class_weights), optimizer='adam')
    HDLTex_lvl1.summary()
    HDLTex_lvl1.fit(X_train, y_train_lvl1.todense(),
                validation_data=(X_test, y_test_lvl1.todense()),
                epochs=epochs,
                verbose=1,
                batch_size=BATCH_SIZE)

    # Train level 2 models
    HDLTex_lvl2 = [None] * number_of_classes_L1
    for i in range(0, number_of_classes_L1):
        if L2_model == 1:
            print('Create CNN submodel of', encoder_lvl1.classes_[i])
            HDLTex_lvl2[i] = buildModel_CNN(word_index, embeddings_index, number_of_classes_L2[i], MAX_SEQUENCE_LENGTH)
        else:
            print('Create RNN submodel of', encoder_lvl1.classes_[i])
            HDLTex_lvl2[i] = buildModel_RNN(word_index, embeddings_index, number_of_classes_L2[i], MAX_SEQUENCE_LENGTH)
        class_weights = calculating_class_weights(y_train_lvl2[L2_train_ind[i]][:,L2_y_col_ind[i]])
        HDLTex_lvl2[i].compile(loss=get_weighted_loss(class_weights), optimizer='adam')
        HDLTex_lvl2[i].fit(X_train[L2_train_ind[i]], y_train_lvl2[L2_train_ind[i]][:,L2_y_col_ind[i]].todense(),
                        validation_data=(X_test[L2_test_ind[i]], y_test_lvl2[L2_test_ind[i]][:,L2_y_col_ind[i]].todense()),
                        epochs=epochs,
                        verbose=1,
                        batch_size=BATCH_SIZE)

    # Train level 3 models
    HDLTex_lvl3 = [None] * len(number_of_classes_L3)
    for i in range(0, len(number_of_classes_L3)):
        if number_of_classes_L3[i] == 1:
            # if a L2 node only has one child node, model cannot be trained on the child category
            # (no negative examples using the exclusive siblings subsetting method)
            continue
        if L3_model == 1:
            print('Create CNN submodel of', encoder_lvl2.classes_[i])
            HDLTex_lvl3[i] = buildModel_CNN(word_index, embeddings_index,number_of_classes_L3[i], MAX_SEQUENCE_LENGTH)
        else:
            print('Create RNN submodel of', encoder_lvl2.classes_[i])
            HDLTex_lvl3[i] = buildModel_RNN(word_index, embeddings_index,number_of_classes_L3[i], MAX_SEQUENCE_LENGTH)
        class_weights = calculating_class_weights(y_train_lvl3[L3_train_ind[i]][:,L3_y_col_ind[i]])
        HDLTex_lvl3[i].compile(loss=get_weighted_loss(class_weights), optimizer='adam')
        HDLTex_lvl3[i].fit(X_train[L3_train_ind[i]], y_train_lvl3[L3_train_ind[i]][:,L3_y_col_ind[i]].todense(),
                        validation_data=(X_test[L3_test_ind[i]], y_test_lvl3[L3_test_ind[i]][:,L3_y_col_ind[i]].todense()),
                        epochs=epochs,
                        verbose=1,
                        batch_size=BATCH_SIZE)

    # Make predictions
    class_weights = calculating_class_weights(y_train_lvl1)
    pred_lvl1 = HDLTex_lvl1.predict(X_test)

    for i in range(0, number_of_classes_L1):
        class_weights = calculating_class_weights(y_train_lvl2[L2_train_ind[i]][:,L2_y_col_ind[i]])
        partial_pred_lvl2 = HDLTex_lvl2[i].predict(X_test)
        if i == 0:
            pred_lvl2 = partial_pred_lvl2
        else:
            pred_lvl2 = np.hstack((pred_lvl2, partial_pred_lvl2))

    for i in range(0, len(number_of_classes_L3)):
        if number_of_classes_L3[i] == 1:
            # If only one label, set 'probability' to 2, such that actual prediction to this category depends entirely on predictions of its ancestor nodes.
            partial_pred_lvl3 = np.array([2]*X_test.shape[0]).reshape(X_test.shape[0],1)
        else:
            class_weights = calculating_class_weights(y_train_lvl3[L3_train_ind[i]][:,L3_y_col_ind[i]])
            partial_pred_lvl3 = HDLTex_lvl3[i].predict(X_test)
        if i == 0:
            pred_lvl3 = partial_pred_lvl3
        else:
            pred_lvl3 = np.hstack((pred_lvl3, partial_pred_lvl3))

    pred_bin_lv1 = np.where(pred_lvl1 > thresh_L1, 1, 0)
    pred_bin_lv2 = np.where(pred_lvl2 > thresh_L2, 1, 0)
    pred_bin_lv3 = np.where(pred_lvl2 > thresh_L3, 1, 0)

    performance = get_performance(y_test_lvl1, y_test_lvl2, y_test_lvl3,
                                  pred_bin_lv1, pred_bin_lv2, pred_bin_lv3,
                                  encoder_lvl1, encoder_lvl2, encoder_lvl3)