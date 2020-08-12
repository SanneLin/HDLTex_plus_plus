# -*- coding: utf-8 -*-
""" This module provides functions for building RNN and CNN models for the HDLTex++ model.

Adapted from https://github.com/kk7nc/HDLTex/blob/master/HDLTex/BuildModel.py

"""

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, GRU

def buildModel_RNN(word_index, embeddings_index, nClasses, max_seq_length):
    """
    Build RNN model.

    Parameters
    ----------
    word_index : dict
        Dictionary mapping words to indices.
    embeddings_index : dict
        Dictionary mapping words to word embeddings.
    nClasses : int
        Number of classes.
    max_seq_length : int
        Maximum length of text sequences.

    Returns
    -------
    model : tensorflow Sequential object
        RNN model.

    """

    _dim = 100
    model = Sequential()
    embedding_matrix = np.zeros((len(word_index) + 1, _dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                _dim,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                trainable=False))
    model.add(GRU(100, dropout=0.25, recurrent_dropout=0.25))
    model.add(Dense(nClasses, activation='sigmoid'))
    return model

def buildModel_CNN(word_index, embeddings_index, nClasses, max_seq_length):
    """
    Create CNN model.

    Parameters
    ----------
    word_index : dict
        Dictionary mapping words to indices.
    embeddings_index : dict
        Dictionary mapping words to word embeddings.
    nClasses : int
        Number of classes.
    max_seq_length : int
        Maximum length of text sequences.

    Returns
    -------
    model : tensorflow Model object
        CNN model.

    """

    _dim = 100
    embedding_matrix = np.zeros((len(word_index) + 1, _dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,
                                _dim,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                trainable=False)

    sequence_input = Input(shape=(max_seq_length,))
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [3, 4, 5, 6, 7]

    for fsz in filter_sizes:
        l_conv = Conv1D(128, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(nClasses, activation='sigmoid')(l_dense)
    model = Model(sequence_input, preds)

    return model