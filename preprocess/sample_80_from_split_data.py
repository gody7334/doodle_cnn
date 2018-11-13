import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import ipdb
import pandas as pd
import numpy as np
from pprint import pprint as pp
from utils.mongodb_conn import *
from utils.mongodb_conn import _connect_mongo
from sklearn.model_selection import RepeatedStratifiedKFold

path_train = '../../data/train_simplified/'
path_CV = '../../data/train_key_id_CV/words/'
path_CV_all = '../../data/train_key_id_CV/all/'
path_CV_all_80 = '../../data/train_key_id_CV/all_80/'
train_files = next(os.walk(path_train))[2]

def sample_80_from_split_dataset():
    for cv_idx in range(10):
        all_train = pd.DataFrame()
        all_val = pd.DataFrame()
        print(cv_idx)
        for tf in train_files:
            tf = tf.replace('.csv','')
            print(tf)
            train = pd.read_csv(f'{path_CV}/train_df_{tf}_{cv_idx}.csv')
            val = pd.read_csv(f'{path_CV}/val_df_{tf}_{cv_idx}.csv')
            train = train.drop(columns=['recognized','word'])
            val = val.drop(columns=['recognized','word'])
            val = val.sample(frac=1.0)
            val_80 = val.iloc[:80]
            train = train.append(val.iloc[80:])
            all_train = all_train.append(train)
            all_val = all_val.append(val_80)

        all_train.to_csv(f'{path_CV_all_80}/train_df_{cv_idx}.csv',index=False)
        all_val.to_csv(f'{path_CV_all_80}/val_df_{cv_idx}.csv',index=False)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    sample_80_from_split_dataset()
