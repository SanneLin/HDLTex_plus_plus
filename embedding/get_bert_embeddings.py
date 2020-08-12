# -*- coding: utf-8 -*-
"""
Convert strings of text into pooled BERT word embeddings.

This script uses bert-as-service to convert strings of text into pooled BERT word embeddings
using the mean or the maximum. For more information, see https://github.com/hanxiao/bert-as-service.

"""

import os
import pickle
import pandas as pd
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient

if __name__ == "__main__":
    # Set path
    TEXT_DATA_DIR = 'PATH TO DATA DIRECTORY'
    BERT_PATH = 'PATH TO BERT MODEL'

    # Load data
    df_train = pd.read_pickle(os.path.join(TEXT_DATA_DIR, 'df_train.pkl'))
    df_val = pd.read_pickle(os.path.join(TEXT_DATA_DIR, 'df_val.pkl'))
    df_test = pd.read_pickle(os.path.join(TEXT_DATA_DIR, 'df_test.pkl'))

    for df in (df_train, df_val, df_test):
        df['text'] = df['EN-title'].str.cat(df['EN-abstract'], sep=' ', na_rep='')


    for i in ('mean', 'max'):
        pooling = 'REDUCE_' + i.upper()

        # Uncomment '-cpu' to use CPU instead of GPU
        args = get_args_parser().parse_args(['-model_dir', BERT_PATH,
                                             '-port', '5555',
                                             '-port_out', '5556',
                                             '-max_seq_len', '512',
                                             # '-cpu',
                                             '-pooling_strategy', pooling])
        server = BertServer(args)
        server.start()
        bc = BertClient()

        # Convert data to pooled BERT embeddings
        BERT_train = bc.encode(df_train['text'].tolist())
        BERT_val = bc.encode(df_val['text'].tolist())
        BERT_test = bc.encode(df_test['text'].tolist())

        # Save embeddings
        with open(os.path.join(TEXT_DATA_DIR, 'BERT_'+i+'_train.pkl'), 'wb') as f:
            pickle.dump(BERT_train,f)
        with open(os.path.join(TEXT_DATA_DIR, 'BERT_'+i+'_val.pkl'), 'wb') as f:
            pickle.dump(BERT_val,f)
        with open(os.path.join(TEXT_DATA_DIR, 'BERT_'+i+'_test.pkl'), 'wb') as f:
            pickle.dump(BERT_test,f)

        bc.close()
        BertServer.shutdown(port=5555)