# -*- coding: utf-8 -*-
""" This module provides functions for loading data. """

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from scipy.sparse import vstack

# Set paths
TEXT_DATA_DIR = 'PATH TO DATA DIRECTORY'
GLOVE_DIR = 'PATH TO GLOVE DIRECTORY'

def load_instances(holdout=True):
    """
    Loads data instances.

    Parameters
    ----------
    holdout : bool, optional
        Loads the train and val samples separately when True,
        the train and validation samples (combined) and test samples when False.
        The default is True.

    Returns
    -------
    train : list of str
        Training samples.
    test : list of str
        Test samples.

    """
    df_train = pd.read_pickle(os.path.join(TEXT_DATA_DIR, 'df_train.pkl'))
    df_val = pd.read_pickle(os.path.join(TEXT_DATA_DIR, 'df_val.pkl'))

    if holdout:
        for df in (df_train, df_val):
            df['text'] = df['EN-title'].str.cat(df['EN-abstract'], sep=' ', na_rep='')
        train, test = df_train['text'].tolist(), df_val['text'].tolist()
    else:
        df_train = df_train.append(df_val, ignore_index=True)
        df_test = pd.read_pickle(os.path.join(TEXT_DATA_DIR, 'df_test.pkl'))
        for df in (df_train, df_test):
            df['text'] = df['EN-title'].str.cat(df['EN-abstract'], sep=' ', na_rep='')
        train, test = df_train['text'].tolist(), df_test['text'].tolist()
    return train, test

def load_bert_instances(holdout=True, pooling='mean'):
    """
    Load BERT embeddings of text data.

    Parameters
    ----------
    holdout : bool, optional
        Loads the train and val samples separately when True,
        the train and validation samples (combined) and test samples when False.
        The default is True.
    pooling : str, optional
        Pooling strategy used for the aggregated word embeddings. The default is 'mean'.

    Raises
    ------
    ValueError
        If pooling is neither 'mean' nor 'max'.

    Returns
    -------
    train : NumPy array
        Training samples.
    val or test : NumPy array
        Test samples.

    """
    if pooling not in ('mean', 'max'):
        raise ValueError('Pooling strategy must be either "mean" or "max".')

    with open(os.path.join(TEXT_DATA_DIR, 'BERT_'+pooling+'_train.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(TEXT_DATA_DIR, 'BERT_'+pooling+'_val.pkl'), 'rb') as f:
        val = pickle.load(f)

    if holdout:
        return train, val
    else:
        with open(os.path.join(TEXT_DATA_DIR, 'BERT_'+pooling+'_test.pkl'), 'rb') as f:
            test = pickle.load(f)
        train = np.concatenate((train, val))
        return train, test

def load_ind_matrices(holdout=True, level=3):
    """
    Loads indicator matrices and corresponding encoder for a single level in the hierarchy.

    Parameters
    ----------
    holdout : bool, optional
        Loads the train and val samples separately when True,
        the train and validation samples (combined) and test samples when False.
        The default is True.
    level : int, optional
        Level of the indicator matrices. 1 for primary level indicator matrices,
        2 for secondary level indicator matrices, 3 for tertiary level label
        matrices. The default is 3.

    Raises
    ------
    ValueError
        If level is not an int between 1 and 3.

    Returns
    -------
    y_train : sparse matrix
        Indicator matrix for the train (or train+validation) samples.
    y_val or y_test: sparse matrix
        Indicator matrix for the validation or test samples.
    encoder : MultiLabelBinarizer object
        Encoder that can convert the indicator matrices to sets of labels and vice cersa.

    """
    if level not in (1, 2, 3):
        raise ValueError('Level must be integer between 1 and 3.')

    with open(os.path.join(TEXT_DATA_DIR, 'y_train_lvl'+str(level)+'.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(TEXT_DATA_DIR, 'y_val_lvl'+str(level)+'.pkl'), 'rb') as f:
        y_val = pickle.load(f)

    encoder = joblib.load(os.path.join(TEXT_DATA_DIR, 'enc_lvl'+str(level)+'.joblib'))

    if holdout:
        return y_train, y_val, encoder
    else:
        y_train = vstack([y_train, y_val])
        with open(os.path.join(TEXT_DATA_DIR, 'y_test_lvl'+str(level)+'.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        return y_train, y_test, encoder


def load_all_ind_matrices(holdout=True):
    """
    Loads indicator matrices of all three levels and corresponding MultiLabelBinarizers.

    Parameters
    ----------
    holdout : bool, optional
        Loads the train and val samples separately when True,
        the train and validation samples (combined) and test samples when False.
        The default is True.

    Returns
    -------
    y_train_lvl1, y_train_lvl2, y_train_lvl3 : sparse matrix
        Primary, secondary, and tertiary level indicator matrices for the train (or train+validation) samples.
    y_test_lvl1, y_test_lvl2, y_test_lvl3 : sparse matrix
        Primary, secondary, and tertiary level indicator matrices for the validation or test samples.
    encoder_lvl1, encoder_lvl2, encoder_lvl3 : MultiLabelBinarizer object
        Multilabel encoders for the primary, secondary, and tertiary level labels.

    """

    y_train_lvl1, y_test_lvl1, encoder_lvl1 = load_ind_matrices(holdout, level=1)
    y_train_lvl2, y_test_lvl2, encoder_lvl2 = load_ind_matrices(holdout, level=2)
    y_train_lvl3, y_test_lvl3, encoder_lvl3 = load_ind_matrices(holdout, level=3)

    return (y_train_lvl1, y_train_lvl2, y_train_lvl3,
            y_test_lvl1, y_test_lvl2, y_test_lvl3,
            encoder_lvl1, encoder_lvl2, encoder_lvl3)

def load_data(*, holdout=True, bert=False, tertiary_only=False, pooling=None):
    """
    Wrapper function for loading data.

    This function loads
        1. data instances, either as text or as BERT embeddings,
        2. indicator matrices, either for the tertiary level only or for all three levels,
        3. corresponding encoders.

    Parameters
    ----------
    holdout : bool, optional
        Loads the train and val samples separately when True,
        the train and validation samples (combined) and test samples when False.
        The default is True.
    bert : bool, optional
        True if BERT embeddings should be loaded, False if text is loaded. The default is False.
    tertiary_only : bool, optional
        True if only the indicator matrices and encoder of the tertiary level should be loaded,
        False otherwise. The default is False.
    pooling : str, optional
        Used to select which aggregated BERT embeddings to load. Only used when
        bert is True. The default is None.

    Returns
    -------
    train : list or NumPy array
        Training samples.
    test : list or NumPy array
        Test samples.
    y_train_lvl1, y_train_lvl2, y_train_lvl3 : sparse matrix
        Primary, secondary, and tertiary level indicator matrices for the train (or train+validation) samples.
    y_test_lvl1, y_test_lvl2, y_test_lvl3 : sparse matrix
        Primary, secondary, and tertiary level indicator matrices for the validation or test samples.
    encoder_lvl1, encoder_lvl2, encoder_lvl3 : MultiLabelBinarizer object
        Multilabel encoders for the primary, secondary, and tertiary level labels.

    """
    if bert:
        train, test = load_bert_instances(holdout, pooling)
    else:
        train, test = load_instances(holdout)

    if tertiary_only:
        y_train_lvl3, y_test_lvl3, encoder_lvl3 = load_ind_matrices(holdout, level=3)
        return train, test, y_train_lvl3, y_test_lvl3, encoder_lvl3
    else:
        (y_train_lvl1, y_train_lvl2, y_train_lvl3,
         y_test_lvl1, y_test_lvl2, y_test_lvl3,
         encoder_lvl1, encoder_lvl2, encoder_lvl3) = load_all_ind_matrices(holdout)
        return (train, test, y_train_lvl1, y_train_lvl2, y_train_lvl3,
                y_test_lvl1, y_test_lvl2, y_test_lvl3, encoder_lvl1,
                encoder_lvl2, encoder_lvl3)

def load_subsets(holdout=True):
    """
    Loads indicator matrices and creates lists of observation indices for each primary and secondary category.

    This function loads the indicator matrices and MultiLabelBinarizers. For each primary and
    secondary category, four lists are created containing the following:
        1. row indices of training instances belonging to the category's descendants,
        2. row indices of test instances belonging to the category's descendants,
        3. names of the category's child categories,
        4. column indices corresponding to the child categories in the indicator matrix.

    Partially adapted from https://github.com/kk7nc/HDLTex/blob/master/HDLTex/Data_helper.py

    Parameters
    ----------
    holdout : bool, optional
        Loads the train and val samples separately when True,
        the train and validation samples (combined) and test samples when False.
        The default is True.

    Returns
    -------
    y_train_lvl1, y_train_lvl2, y_train_lvl3 : sparse matrix
        Primary, secondary, and tertiary level indicator matrices for the train (or train+validation) samples.
    y_test_lvl1, y_test_lvl2, y_test_lvl3 : sparse matrix
        Primary, secondary, and tertiary level indicator matrices for the validation or test samples.
    encoder_lvl1, encoder_lvl2, encoder_lvl3 : MultiLabelBinarizer object
        Multilabel encoders for the primary, secondary, and tertiary level labels.
    number_of_classes_L1 : int
        Number of primary categories.
    number_of_classes_L2 : list of int
        Number of secondary categories per primary category.
    number_of_classes_L3 : list of int
        Number of tertiary categories per secondary category.
    L2_train_ind : list of list of int
        Contains for each primary category the list of row indices of training samples belonging
        to the category's descendant categories.
    L2_test_ind : list of list of int
        Contains for each primary category the list of row indices of test samples belonging
        to the category's descendant categories.
    L2_y_labels : list of list of str
        Contains for each primary category the list of child categories.
    L2_y_col_ind : list of list of int
        Contains for each primary category the list of column indices of y_train_lvl2 which
        correspond to the category's child categories.
    L3_train_ind : list of list of int
        Contains for each secondary category the list of row indices of training samples belonging
        to the category's child categories.
    L3_test_ind : list of list of int
        Contains for each secondary category the list of row indices of test samples belonging
        to the category's child categories..
    L3_y_labels : list of list of str
        Contains for each secondary category the list of child categories.
    L3_y_col_ind : list of list of int
        Contains for each secondary category the list of column indices of y_train_lvl3 which
        correspond to the category's child categories.

    """
    (y_train_lvl1, y_train_lvl2, y_train_lvl3,
     y_test_lvl1, y_test_lvl2, y_test_lvl3,
    encoder_lvl1, encoder_lvl2, encoder_lvl3) = load_all_ind_matrices(holdout)

    number_of_classes_L1 = len(encoder_lvl1.classes_)
    total_L2_classes = len(encoder_lvl2.classes_)
    number_of_classes_L2 = np.zeros(number_of_classes_L1, dtype=int)

    L2_train_ind = [[] for _ in range(number_of_classes_L1)]
    L2_test_ind = [[] for _ in range(number_of_classes_L1)]
    L2_y_labels = [[] for _ in range(number_of_classes_L1)]
    L2_y_col_ind = [[] for _ in range(number_of_classes_L1)]

    # For each primary category, create list of indices of observations
    # belonging to the descendants of the category
    for (row, col) in zip(*y_train_lvl1.nonzero()):
        L2_train_ind[col].append(row)
    for (row, col) in zip(*y_test_lvl1.nonzero()):
        L2_test_ind[col].append(row)

    # Count the number of secondary categories per primary category
    for ind, label in enumerate(encoder_lvl1.classes_):
        number_of_classes_L2[ind] = len([i for i in encoder_lvl2.classes_ if i[0]==label])

    # Store secondary categories and their corresponding column numbers for each primary category
    cml_sum = 0
    for ind, num_labels in enumerate(number_of_classes_L2):
        L2_y_labels[ind] = encoder_lvl2.classes_[cml_sum:(cml_sum + num_labels)]
        L2_y_col_ind[ind] = list(range(cml_sum, (cml_sum + num_labels)))
        cml_sum += num_labels

    number_of_classes_L3 = np.zeros(total_L2_classes, dtype=int)

    L3_train_ind = [[] for _ in range(total_L2_classes)]
    L3_test_ind = [[] for _ in range(total_L2_classes)]
    L3_y_labels = [[] for _ in range(total_L2_classes)]
    L3_y_col_ind = [[] for _ in range(total_L2_classes)]

    for (row, col) in zip(*y_train_lvl2.nonzero()):
        L3_train_ind[col].append(row)
    for (row, col) in zip(*y_test_lvl2.nonzero()):
        L3_test_ind[col].append(row)

    for ind, label in enumerate(encoder_lvl2.classes_):
        number_of_classes_L3[ind] = len([i for i in encoder_lvl3.classes_ if i[0:2]==label])

    cml_sum = 0
    for ind, num_labels in enumerate(number_of_classes_L3):
        L3_y_labels[ind] = encoder_lvl3.classes_[cml_sum:(cml_sum + num_labels)]
        L3_y_col_ind[ind] = list(range(cml_sum, (cml_sum + num_labels)))
        cml_sum += num_labels

    return (y_train_lvl1, y_train_lvl2, y_train_lvl3,
            y_test_lvl1, y_test_lvl2, y_test_lvl3,
            encoder_lvl1, encoder_lvl2, encoder_lvl3,
            number_of_classes_L1, number_of_classes_L2, number_of_classes_L3,
            L2_train_ind, L2_test_ind, L2_y_labels, L2_y_col_ind,
            L3_train_ind, L3_test_ind, L3_y_labels, L3_y_col_ind)

def load_data_and_tokenize(holdout=True, max_nb_words=None):
    """
    Loads data instances and tokenizes each instance.

    Parameters
    ----------
    holdout : bool, optional
        Loads the train and val samples separately when True,
        the train and validation samples (combined) and test samples when False.
        The default is True.
    max_nb_words : int, optional
        The maximum number of words to keep, based
        on word frequency. Only the most common `max_nb_words-1` words will
        be kept. The default is None.

    Returns
    -------
    sequences_train : list of list of int
        List containing a sequence of integers (indices) for each training observation.
    sequences_test : list of list of int
        List containing a sequence of integers (indices) for each test observation.
    word_index : dict
        Dictionary mapping words to their indices.

    """
    train_text, test_text = load_instances(holdout)

    glove_tokenizer = Tokenizer(num_words=max_nb_words)
    glove_tokenizer.fit_on_texts(train_text)
    sequences_train = glove_tokenizer.texts_to_sequences(train_text)
    sequences_test = glove_tokenizer.texts_to_sequences(test_text)
    word_index = glove_tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    return sequences_train, sequences_test, word_index

def load_glove():
    """
    Load pre-trained word embeddings into dictionary.

    Returns
    -------
    embeddings_index : dict
        Dictionary mapping words to their embeddings.

    """
    embeddings_index = {}
    Glove_path = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
    print(Glove_path)
    with open(Glove_path, 'r', encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index