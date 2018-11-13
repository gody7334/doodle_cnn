
# coding: utf-8

# In[1]:


# add parent folder as root folder
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import pandas as pd
import numpy as np
from utils.mongodb_conn import *
from utils.mongodb_conn import _connect_mongo
from joblib import Parallel, delayed


# In[2]:


path_train = '../../data/train_simplified/'
train_files = next(os.walk(path_train))[2]


# In[ ]:


for tf in train_files:
    print(tf)
    df = pd.read_csv(f"{path_train}{tf}")
    df_dict = df.to_dict('records')

    conn = _connect_mongo(host='localhost', port=27017, username=None, password=None, db='dataset')
    conn.doodle_quickdraw.insert_many(df_dict)

#     Parallel(n_jobs=8)(delayed(insert_data)(data, 'dataset', 'doodle_quickdraw', 'key_id') for data in df_dict)

#     for data in df_dict:
#          insert_data(data, 'dataset', 'doodle_quickdraw', 'key_id')
#     os.exit()


# In[25]:


# df_dict[0]

