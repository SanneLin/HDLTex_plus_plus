# -*- coding: utf-8 -*-
""" This module can be used to convert strings of text to a pooled embedding vector.

Adapted from https://www.kaggle.com/nhrade/text-classification-using-word-embeddings
"""

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin

# Set paths
TEXT_DATA_DIR = 'PATH TO DATA DIRECTORY'
GLOVE_DIR = 'PATH TO GLOVE DIRECTORY'

class GloVeEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Class that transforms strings of text into pooled GloVe embeddings using the mean or the maximum.

    ...

    Attributes
    ----------
    _embedding_dim : int
        length of word embeddings.
    pooling : str
        pooling strategy, either 'mean' or 'max'.
    tokenizer : Tokenizer object
        tokenizer for converting strings into tokens.
    index_word : dict
        dictionary mapping indices to words.
    _E : dict
        dictionary mapping words to embeddings.

    """

    _embedding_dim = 100

    def __init__(self, pooling='mean'):
        """
        Constructor for the GloVeEmbeddingTransformer class.

        Parameters
        ----------
        pooling : str, optional
            pooling strategy used for aggregating embeddings. Should be
            either 'mean' or 'max'. The default is 'mean'.

        Raises
        ------
        ValueError
            If the pooling strategy is neither 'mean' nor 'max'.

        Returns
        -------
        None.

        """
        if pooling not in ('mean', 'max'):
            raise ValueError('Pooling strategy must be either "mean" or "max".')
        self.pooling = pooling
        self.tokenizer = Tokenizer()
        self.index_word = None
        self._E = self._load_words()

    def _load_words(self):
        print('Indexing word vectors.')
        embeddings_index = {}
        Glove_path = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
        with open(Glove_path, 'r', encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def _get_sentence_emb(self, doc_sequence):
        if self.pooling not in ('mean', 'max'):
            raise ValueError('Pooling strategy must be either "mean" or "max".')

        embedding_matrix = []
        for token in doc_sequence:
            embedding_vector = self._E.get(self.index_word.get(token))
            if embedding_vector is not None:
                embedding_matrix.append(embedding_vector)

        if len(embedding_matrix) == 0:
            return np.zeros(GloVeEmbeddingTransformer._embedding_dim)
        if self.pooling == 'max':
            return np.max(np.array(embedding_matrix), axis=0)
        else:
            return np.mean(np.array(embedding_matrix), axis=0)

    def fit(self, X, y=None):
        """
        Wrapper function that fits the tokenizer using input text.

        Parameters
        ----------
        X : list of str or NumPy array of str
            text used for fitting the tokenizer.
        y : object, optional
            ignored.

        Returns
        -------
        GloVeEmbeddingTransformer object
            reference to the instance object.

        """
        self.tokenizer.fit_on_texts(X)
        self.index_word = {v: k for k, v in self.tokenizer.word_index.items()}
        print('Found %s unique tokens.' % len(self.tokenizer.word_index))
        return self

    def transform(self, X):
        """
        Transforms text into a pooled word embeddings.

        Parameters
        ----------
        X : list of str or NumPy array
            text to be converted into pooled word embeddings.

        Returns
        -------
        NumPy array
            array of pooled word embeddings.

        """
        sequences = self.tokenizer.texts_to_sequences(X)
        return np.array([self._get_sentence_emb(doc) for doc in sequences])

    def fit_transform(self, X, y=None):
        """
        Wrapper function that fits then transforms the data.

        Parameters
        ----------
        X : list of str or NumPy array
            text to be converted into pooled word embeddings.
        y : object, optional
            ignored.

        Returns
        -------
        X_new : ndarray array
            Transformed array.

        """
        return self.fit(X).transform(X)