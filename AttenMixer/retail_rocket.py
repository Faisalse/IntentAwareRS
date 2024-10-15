# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 08:57:29 2024

@author: shefai
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
SESSION_LENGTH = 30 * 60 #30 minutes
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#min date config
MIN_DATE = '2014-04-01'

#days test default config 
DAYS_TEST = 7


def load_data_retail( file):
    #load csv
    data = pd.read_csv( file, sep=',', header=0, usecols=[0,1,2,3], dtype={0:np.int64, 1:np.int32, 2:str, 3:np.int32})
    #specify header names
    data.columns = ['Time','UserId','Type','ItemId']
    data['Time'] = (data.Time / 1000).astype( int )
    #sessionize
    data.sort_values(by=['UserId', 'Time'], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data['Time'].values)
    # check which of them are bigger then session_th
    split_session = tdiff > SESSION_LENGTH
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data['UserId'].values[1:] != data['UserId'].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['SessionId'] = session_ids
    data.sort_values( ['SessionId','Time'], ascending=True, inplace=True )
    
    itemslist = list(data["ItemId"])
    catelist = []
    count = 0
    return data

def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) : 
    
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>1 ].index)]
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;

def split_data_only( data, days_test = DAYS_TEST):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-(86400 * days_test)].index
    session_test = session_max_times[session_max_times >= tmax-(86400 * days_test)].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    return train, test

def validation_data( data):
    train_val, test_val = split_data_only( data)
    return train_val, test_val

def data_augmentation_transformation_for_Dl(train, test):
    session_key = "SessionId"
    item_key = "ItemId"
    index_session = train.columns.get_loc( session_key)
    index_item = train.columns.get_loc( item_key )
    
    session_item_train = {}
    # Convert the session data into sequence
    for row in train.itertuples(index=False):
        if row[index_session] in session_item_train:
            session_item_train[row[index_session]] += [(row[index_item])] 
        else: 
            session_item_train[row[index_session]] = [(row[index_item])]
   
    word2index ={}
    index2wiord = {}
    item_no = 1
    for key, values in session_item_train.items():
        length = len(session_item_train[key])
        for i in range(length):
            if session_item_train[key][i] in word2index:
                session_item_train[key][i] = word2index[session_item_train[key][i]]
            else:
                word2index[session_item_train[key][i]] = item_no
                index2wiord[item_no] = session_item_train[key][i]
                session_item_train[key][i] = item_no
                item_no +=1
                
                
    
    features_train = []
    targets_train = []
    for value in session_item_train.values():
        for i in range(1, len(value)):
            targets_train.append(value[-i])
            features_train.append(value[:-i])
            
            
    session_item_test = {}
    # Convert the session data into sequence
    for row in test.itertuples(index=False):
        if row[index_session] in session_item_test:
            session_item_test[row[index_session]] += [(row[index_item])] 
        else: 
            session_item_test[row[index_session]] = [(row[index_item])]
            
    for key, values in session_item_test.items():
        length = len(session_item_test[key])
        for i in range(length):
            if session_item_test[key][i] in word2index:
                session_item_test[key][i] = word2index[session_item_test[key][i]]
            else:
                word2index[session_item_test[key][i]] = item_no
                index2wiord[item_no] = session_item_test[key][i]
                session_item_test[key][i] = item_no
                item_no +=1
    
    features_test = []
    targets_test = []
    for value in session_item_test.values():
        for i in range(1, len(value)):
            targets_test.append(value[-i])
            features_test.append(value[:-i])
    
            
    item_no = item_no +1
    return [features_train, targets_train], [features_test, targets_test], item_no
