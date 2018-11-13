
# coding: utf-8

# In[1]:


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


# In[2]:


path_train = '../../data/train_simplified/'
path_CV = '../../data/train_key_id_CV/'
train_files = next(os.walk(path_train))[2]


# In[7]:


df = sample_mongo(
    db='dataset',
    collection='doodle_quickdraw',
    query={},
    project={'key_id':1, 'recognized':1, 'word':1},
    host='localhost',
    port=27017,
    username=None,
    password=None,
    no_id=True,
    num_sample=10000
)

df = df.loc[df['recognized'] == True].reset_index()

ids = df.index.values
word_class = df.word.values

rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1,random_state=999)

cv_idx = 0
for train_idx, test_idx in rskf.split(ids, word_class):
    train_cv = df.loc[train_idx]
    test_cv = df.loc[test_idx]
    sys.exit()
    train_cv.to_csv(f'{path_CV}/train_df_{cv_idx}.csv')
    test_cv.to_csv(f'{path_CV}/val_df_{cv_idx}.csv')
    cv_idx+=1


# In[8]:


train_cv

