# -*- coding: utf-8 -*-
""" Main code for cleaning and splitting data. """

import pandas as pd
import os
import string
import pickle
import fasttext
import joblib
from text_cleaning import clean_string, clean_abstract, extract_english, clean_jel, remove_symbols
from train_test_split import iterative_train_test
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == "__main__":

    # Set paths
    FASTTEXT_DIR = 'PATH TO FASTTEXT MODEL'
    TEXT_DATA_DIR = 'PATH TO DATA DIRECTORY'

    data_file = os.path.join(TEXT_DATA_DIR, 'NAME OF DATA FILE')
    model = fasttext.load_model(os.path.join(FASTTEXT_DIR, 'lid.176.bin'))
    jel_codes_file = 'NAME OF FILE CONTAINING ALL JEL CODES'

    # Load and clean data
    df = pd.read_csv(data_file)

    with open(jel_codes_file) as f:
        jel_labels = [line.split(',') for line in f]
        jel_labels = [item.strip() for sublist in jel_labels for item in sublist]

    df['title'] = df['title'].astype(str).apply(clean_string)
    df['abstract'] = df['abstract'].astype(str).apply(clean_abstract, fasttext_model=model)
    df = df[(df['title'] != '') & (df['abstract'] != '')]

    df['EN-title'] = df['title'].apply(extract_english, fasttext_model=model)
    df['EN-abstract'] = df['abstract'].apply(extract_english, fasttext_model=model)
    df = df[(df['EN-title'] != '') & (df['EN-abstract'] != '')]

    df['EN-title'] = df['EN-title'].apply(remove_symbols, keep=set(string.printable).difference('"'))
    df['EN-abstract'] = df['EN-abstract'].apply(remove_symbols, keep=set(string.printable).difference('"'))
    df = df[(df['EN-title'] != '') & (df['EN-abstract'] != '')]

    df['JEL'] = df['classification-jel'].astype(str).apply(clean_jel, labels=jel_labels)

    # Keep rows with JEL codes
    df = df.loc[(df['JEL'].map(len) > 0), ['EN-title', 'EN-abstract', 'JEL']]

    # Combine (exact) duplicates
    df = df.groupby([df['EN-title'].str.lower(),
                     df['EN-abstract'].str.lower()],
                    as_index=False).agg({'EN-title': 'first', 'EN-abstract': 'first',
                                         'JEL': lambda x: set.union(*x)}).reset_index(drop=True)

    # Create indicator matrix, where (i,j) is 1 if observation i belongs to category j and 0 otherwise
    enc = MultiLabelBinarizer(sparse_output=True)
    y_data = enc.fit_transform(df['JEL'])

    # Split data twice to obtain train, val, and test sets
    X_train_large, y_train_large, X_test, y_test = iterative_train_test(df, y_data, test_size=0.2)
    X_train, y_train, X_val, y_val = iterative_train_test(X_train_large, y_train_large, test_size=0.2)

    # Create new MultiLabelBinarizer, fit using the dataset with the fewest unique labels
    # (such that labels not occuring in all three datasets are removed)
    enc_val = MultiLabelBinarizer(sparse_output=True)
    enc_val.fit(X_val['JEL'])

    y_train_small2 = enc_val.transform(X_train['JEL'])
    y_test2 = enc_val.transform(X_test['JEL'])
    y_val2 = enc_val.transform(X_val['JEL'])

    joblib.dump(enc_val, os.path.join(TEXT_DATA_DIR, 'enc_lvl3.joblib'))

    # Create indicator matrices of primary and secondary level labels
    for X in (X_train, X_val, X_test):
        X['JEL_lvl1'] = X['JEL'].apply(lambda x: sorted({i[:1] for i in x}))
        X['JEL_lvl2'] = X['JEL'].apply(lambda x: sorted({i[:2] for i in x}))

    enc_val_lvl1 = MultiLabelBinarizer(sparse_output=True)
    enc_val_lvl1.fit(X_val['JEL_lvl1'])

    y_train_lvl1 = enc_val_lvl1.transform(X_train['JEL_lvl1'])
    y_val_lvl1 = enc_val_lvl1.transform(X_val['JEL_lvl1'])
    y_test_lvl1 = enc_val_lvl1.transform(X_test['JEL_lvl1'])

    joblib.dump(enc_val_lvl1, os.path.join(TEXT_DATA_DIR, 'enc_lvl1.joblib'))

    enc_val_lvl2 = MultiLabelBinarizer(sparse_output=True)
    enc_val_lvl2.fit(X_val['JEL_lvl2'])

    y_train_lvl2 = enc_val_lvl2.transform(X_train['JEL_lvl2'])
    y_val_lvl2 = enc_val_lvl2.transform(X_val['JEL_lvl2'])
    y_test_lvl2 = enc_val_lvl2.transform(X_test['JEL_lvl2'])

    joblib.dump(enc_val_lvl2, os.path.join(TEXT_DATA_DIR, 'enc_lvl2.joblib'))

    # Save datasets
    with open(os.path.join(TEXT_DATA_DIR, 'y_train_lvl1.pkl'), 'wb') as f:
        pickle.dump(y_train_lvl1, f)
    with open(os.path.join(TEXT_DATA_DIR, 'y_test_lvl1.pkl'), 'wb') as f:
        pickle.dump(y_test_lvl1, f)
    with open(os.path.join(TEXT_DATA_DIR, 'y_val_lvl1.pkl'), 'wb') as f:
        pickle.dump(y_val_lvl1, f)

    with open(os.path.join(TEXT_DATA_DIR, 'y_train_lvl2.pkl'), 'wb') as f:
        pickle.dump(y_train_lvl2, f)
    with open(os.path.join(TEXT_DATA_DIR, 'y_test_lvl2.pkl'), 'wb') as f:
        pickle.dump(y_test_lvl2, f)
    with open(os.path.join(TEXT_DATA_DIR, 'y_val_lvl2.pkl'), 'wb') as f:
        pickle.dump(y_val_lvl2, f)

    with open(os.path.join(TEXT_DATA_DIR, 'y_train_lvl3.pkl'), 'wb') as f:
        pickle.dump(y_train_small2, f)
    with open(os.path.join(TEXT_DATA_DIR, 'y_test_lvl3.pkl'), 'wb') as f:
        pickle.dump(y_test2, f)
    with open(os.path.join(TEXT_DATA_DIR, 'y_val_lvl3.pkl'), 'wb') as f:
        pickle.dump(y_val2, f)

    X_train = X_train[['EN-title', 'EN-abstract', 'JEL']]
    X_val = X_val[['EN-title', 'EN-abstract', 'JEL']]
    X_test = X_test[['EN-title', 'EN-abstract', 'JEL']]

    X_train.to_pickle(os.path.join(TEXT_DATA_DIR, 'df_train.pkl'))
    X_val.to_pickle(os.path.join(TEXT_DATA_DIR, 'df_val.pkl'))
    X_test.to_pickle(os.path.join(TEXT_DATA_DIR, 'df_test.pkl'))








