import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import pandas as pd
import numpy as np
from pprint import pprint as pp
from utils.mongodb_conn import *
from utils.mongodb_conn import _connect_mongo
from sklearn.model_selection import RepeatedStratifiedKFold

path_train = '../../data/train_simplified/'
path_CV = '../../data/train_key_id_CV/words/'
path_CV_all = '../../data/train_key_id_CV/all/'
train_files = next(os.walk(path_train))[2]

def merge_doodle_dataset_each_to_whole_CV():
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
            all_train = all_train.append(train)
            all_val = all_val.append(val)

        all_train.to_csv(f'{path_CV_all}/train_df_{cv_idx}.csv',index=False)
        all_val.to_csv(f'{path_CV_all}/val_df_{cv_idx}.csv',index=False)




def split_doodle_dataset_whole_with_class_stratification():
    for tf in train_files:
        tf = tf.replace('.csv','')
        print(tf)
        df = read_mongo(
            db='dataset',
            collection='doodle_quickdraw',
            query={'word': tf},
            project={'key_id':1, 'recognized':1, 'word':1},
            host='localhost',
            port=27017,
            username=None,
            password=None,
            no_id=True,
            # num_sample=10000
        )

        df = df.loc[df['recognized'] == True].reset_index()

        print('finished df')

        ids = df.index.values
        word_class = df.word.values

        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1,random_state=999)

        print('split ready')

        cv_idx = 0
        for train_idx, test_idx in rskf.split(ids, word_class):
            train_cv = df.loc[train_idx]
            test_cv = df.loc[test_idx]
            train_cv.to_csv(f'{path_CV}/train_df_{tf}_{cv_idx}.csv',index=False)
            test_cv.to_csv(f'{path_CV}/val_df_{tf}_{cv_idx}.csv',index=False)
            cv_idx+=1

def split_doodle_dataset_by_each_file():
    for tf in train_files:
        tf = tf.replace('.csv','')
        print(tf)
        df = read_mongo(
            db='dataset',
            collection='doodle_quickdraw',
            # need to create joined index for and condition,
            # otherwise performance will be terrible....
            # query={'word':tf, 'recognized':{'$ne': True}},
            query={'word':tf},
            project={'key_id':1, 'recognized':1, 'word':1},
            host='localhost',
            port=27017,
            username=None,
            password=None,
            no_id=True,
        )
        df = df.loc[df['recognized'] == True]
        pp(df.head(2))
        os.exit()

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_simple_to_image()
    split_doodle_dataset_whole_with_class_stratification()
    # merge_doodle_dataset_each_to_whole_CV()
