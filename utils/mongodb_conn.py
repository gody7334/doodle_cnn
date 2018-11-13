import pandas as pd
import pymongo
import numpy as np
from pymongo import MongoClient

def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """
    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)
    return conn[db]

conn = None
def insert_data(data, db, collection, check_id='id', host='localhost', port=27017, username=None, password=None, no_id=True):
    global conn
    if conn is None:
        conn = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    duplicate_result = conn[collection].find(
       {check_id: data[check_id]})

#     if duplicate_result.count() > 0:
#         print('duplicate count' + str(duplicate_result.count()))
    if duplicate_result.count() == 0:
        conn[collection].insert_one(data)

def set_data(db, collection, search, set_query, host='localhost', port=27017, username=None, password=None, no_id=True):
    global conn
    if conn is None:
        conn = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    conn[collection].update(
        search,
        { '$set': set_query }
    )

def read_mongo(db, collection, query={}, project={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    global conn
    if conn is None:
        conn = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    if not project:
        cursor = conn[collection].find(query)
    else:
        cursor = conn[collection].find(query,project)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df

def sample_mongo(db,
        collection,
        query={},
        project={},
        host='localhost',
        port=27017,
        username=None,
        password=None,
        no_id=True,
        num_sample=1000):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    global conn
    if conn is None:
        conn = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    if not project:
        cursor = conn[collection].aggregate([{ "$match": query},{ "$sample": { "size": num_sample }}])
    else:
        cursor = conn[collection].aggregate([{ "$match": query},{ "$sample": { "size": num_sample }},{ "$project": project }])

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df

