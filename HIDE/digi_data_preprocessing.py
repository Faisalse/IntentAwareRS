# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:25:25 2024

@author: shefai
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import csv
import time
import pickle
import operator
DATA_FILE = 'train-item-views'
from datetime import datetime

COLS = [0, 2, 3, 4]
# days test default config
DAYS_TEST = 3
MINIMUM_ITEM_SUPPORT = 5
MINIMUM_SESSION_LENGTH = 2

# preprocessing from original gru4rec -  uses just the last day as test
    
def load_data(file):
    timestampp = list()
    dataset = pd.read_csv(file, sep =";")
    dataset = dataset.iloc[:400000, :]
    
    
    with open(file, "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        for data in reader:  
            Time = time.mktime(time.strptime(data['eventdate'], '%Y-%m-%d'))
            timestampp.append(Time)   
            
            if len(timestampp) == len(dataset):
                break
        dataset["Time"] = timestampp
    # change the columns name...
    dataset.rename(columns = {"sessionId":"SessionId", 'itemId':'ItemId'}, inplace = True)   
    del dataset["userId"]
    del dataset["timeframe"]
    del dataset["eventdate"]
    
    
    return dataset

def filter_data(data, min_item_support= MINIMUM_ITEM_SUPPORT, min_session_length= MINIMUM_SESSION_LENGTH):
    # filter session length
    session_lengths = data.groupby('SessionId').size()
    session_lengths = session_lengths[ session_lengths >= min_session_length ]
    data = data[np.in1d(data.SessionId, session_lengths.index)]
    
    # filter item support
    data['ItemSupport'] = data.groupby('ItemId')['ItemId'].transform('count')
    data = data[data.ItemSupport >= min_item_support]

    # filter session length again, after filtering items
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]
    
    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set default \n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))
    
    del data["ItemSupport"]
    return data;

def split_data(data, days_test = DAYS_TEST):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]
    
    min_date = datetime.fromtimestamp(train.Time.min())
    max_date = datetime.fromtimestamp(train.Time.max())
    
    difference = max_date - min_date
    print("Number of training days:", difference.days)
    
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    
    
    # validation data.....
    
    
    
    #test.to_csv(output_file + 'test.txt', sep='\t', index=False)
    
    min_date = datetime.fromtimestamp(test.Time.min())
    max_date = datetime.fromtimestamp(test.Time.max())
    
    difference = max_date - min_date
    print("Number of testing days:", difference.days)
    
    # create data strucutre for GNN models
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
    return (features_train, targets_train), (features_test, targets_test), item_no
    
def split_data_digi_baseline(data, days_test = DAYS_TEST):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]
    
    min_date = datetime.fromtimestamp(train.Time.min())
    max_date = datetime.fromtimestamp(train.Time.max())
    
    difference = max_date - min_date
    print("Number of training days:", difference.days)
    
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    
    
    #train.to_csv("digi_train_full.txt", sep = "\t", index = False)
    #test.to_csv("digi_test.txt", sep = "\t", index = False)
    
    
    # validation data
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)
    
    
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    validation_data_train = train[np.in1d(train.SessionId, session_train)]
    
    min_date = datetime.fromtimestamp(validation_data_train.Time.min())
    max_date = datetime.fromtimestamp(validation_data_train.Time.max())
    
    difference = max_date - min_date
    print("Number of days for validation data:", difference.days)
    
    trlength = validation_data_train.groupby('SessionId').size()
    validation_data_train = validation_data_train[np.in1d(validation_data_train.SessionId, trlength[trlength>=2].index)]
    validation_test = train[np.in1d(train.SessionId, session_test)]
    
    validation_test = validation_test[np.in1d(validation_test.ItemId, validation_data_train.ItemId)]
    tslength = validation_test.groupby('SessionId').size()
    validation_test = validation_test[np.in1d(validation_test.SessionId, tslength[tslength >= 2].index)]
    
    
    print('Full validation tr set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(validation_data_train), validation_data_train.SessionId.nunique(),
                                                                             validation_data_train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Validation test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(validation_test), validation_test.SessionId.nunique(),
    
                                                                       validation_test.ItemId.nunique()))
    
    
    #validation_data_train.to_csv("digi_train_tr.txt", sep = "\t", index = False)
    #validation_test.to_csv("digi_train_valid.txt", sep = "\t", index = False)
    
    
    unique_items_ids = data["ItemId"].unique()
    
    min_date = datetime.fromtimestamp(test.Time.min())
    max_date = datetime.fromtimestamp(test.Time.max())
    
    difference = max_date - min_date
    print("Number of testing days:", difference.days)
    
    
    session_key = "SessionId"
    item_key = "ItemId"
    index_session = test.columns.get_loc( session_key)
    index_item = test.columns.get_loc( item_key )
    session_item_test = {}
    # Convert the session data into sequence
    for row in test.itertuples(index=False):
        if row[index_session] in session_item_test:
            session_item_test[row[index_session]] += [(row[index_item])] 
        else: 
            session_item_test[row[index_session]] = [(row[index_item])]
            
    
    features_test = []
    targets_test = []
    for value in session_item_test.values():
        for i in range(1, len(value)):
            targets_test.append(value[-i])
            features_test.append(value[:-i])
   
    return train, [features_test, targets_test], unique_items_ids 
    
def split_data_digi_knn(data, days_test = DAYS_TEST):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]
    
    min_date = datetime.fromtimestamp(train.Time.min())
    max_date = datetime.fromtimestamp(train.Time.max())
    
    difference = max_date - min_date
    print("Number of training days:", difference.days)
    
    trlength = train.groupby('SessionId').size()
    train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    
    
    train.to_csv("digi_train_full.txt", sep = "\t", index = False)
    test.to_csv("digi_test.txt", sep = "\t", index = False)
    
    
    # validation data
    data_end = datetime.fromtimestamp(train.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)
    
    
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times <= test_from.timestamp()].index
    session_test = session_max_times[session_max_times > test_from.timestamp()].index
    validation_data_train = train[np.in1d(train.SessionId, session_train)]
    
    min_date = datetime.fromtimestamp(validation_data_train.Time.min())
    max_date = datetime.fromtimestamp(validation_data_train.Time.max())
    
    difference = max_date - min_date
    print("Number of days for validation data:", difference.days)
    
    trlength = validation_data_train.groupby('SessionId').size()
    validation_data_train = validation_data_train[np.in1d(validation_data_train.SessionId, trlength[trlength>=2].index)]
    validation_test = train[np.in1d(train.SessionId, session_test)]
    
    validation_test = validation_test[np.in1d(validation_test.ItemId, validation_data_train.ItemId)]
    tslength = validation_test.groupby('SessionId').size()
    validation_test = validation_test[np.in1d(validation_test.SessionId, tslength[tslength >= 2].index)]
    
    
    print('Full validation tr set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(validation_data_train), validation_data_train.SessionId.nunique(),
                                                                             validation_data_train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Validation test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(validation_test), validation_test.SessionId.nunique(),
    
                                                                       validation_test.ItemId.nunique()))
    
    
    validation_data_train.to_csv("digi_train_tr.txt", sep = "\t", index = False)
    validation_test.to_csv("digi_train_valid.txt", sep = "\t", index = False)
    
    
    unique_items_ids = data["ItemId"].unique()
    
    min_date = datetime.fromtimestamp(test.Time.min())
    max_date = datetime.fromtimestamp(test.Time.max())
    
    difference = max_date - min_date
    print("Number of testing days:", difference.days)
    
    
   
    return train, test, unique_items_ids     
    
#
#if __name__ == '__main__':
    #path = "datasets/diginetica/raw/train-item-views.csv"
    #output_file = "datasets/diginetica/process_data/"
    # reality 
    #dataset = load_data(path) 
    
    #filter_data = filter_data(dataset)
    #features_train, targets_train, features_test, targets_test, item_no = split_data(filter_data)




