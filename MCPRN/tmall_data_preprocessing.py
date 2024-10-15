# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:25:25 2024

@author: shefai
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import random
random.seed(42)
from datetime import datetime
COLS = [0, 2, 3, 4]
# days test default config
DAYS = 30
MINIMUM_ITEM_SUPPORT = 3
MINIMUM_SESSION_LENGTH = 2

# preprocessing from original gru4rec -  uses just the last day as test
    
def load_data_tmall( path):     
    
    data = pd.read_csv(path, sep = "\t")
    #data = data[data["SessionId"] < 120000]
    del data["UserId"]


    session_key ='SessionId'
    item_key= 'ItemId'
    time_key='Time'
    
    print("Information about raw data")
    print("Number of sesssions: "+   str( len(data["SessionId"].unique())  ))
    print("Number of Items: "+   str( len(data["ItemId"].unique())  ))
    return data
def filter_data_tmall( data, min_item_support=MINIMUM_ITEM_SUPPORT, min_session_length=MINIMUM_SESSION_LENGTH ) : 
    
    counter = True
    while (counter):
        session_group = data.groupby("SessionId").size()
        session_more_than1 = [key for key, value in session_group.items() if value >= min_session_length]
        data = data[data["SessionId"].isin(session_more_than1)]

        item_group = data.groupby("ItemId").size()
        items_more_than5 = [key for key, value in item_group.items() if value >= min_item_support]
        data = data[data["ItemId"].isin(items_more_than5)]
        session_group = data.groupby("SessionId").size()
        item_group = data.groupby("ItemId").size()
        
        if min(session_group) >= min_session_length and min(item_group) >= min_item_support:
            counter = False
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    return data


def split_list(input_list, split_ratio=0.7):
    # Shuffle the list to randomize the elements
    random.shuffle(input_list)
    # Calculate the index at which to split the list
    split_index = int(len(input_list) * split_ratio)
    # Split the list into two parts
    part_70 = input_list[:split_index]
    part_30 = input_list[split_index:]
    return part_70, part_30

def split_data_tmall( data, tes_days = DAYS):
    print("data spliting phase")
    tmax = data.Time.max()
    # last 30 days data
    last30Days = tmax - (86400 * tes_days)
    data = data[data["Time"] >= last30Days]

    print("Dataset info after selecting last 30 days data")
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )

    min_date = datetime.fromtimestamp(last30Days)
    max_date = datetime.fromtimestamp(data.Time.max())
    difference = max_date - min_date
    print("Number of days:", difference.days)

    # all sessions
    input_list = list(data["SessionId"].unique())
    part_70, part_30 = split_list(input_list, split_ratio=0.7)
    print("Number of training sessions:  "+str(len(part_70)))
    print("Number of testing sessions:  "+str(len(part_30)))
    train = data[data["SessionId"].isin(part_70)]
    test = data[data["SessionId"].isin(part_30)]

    test = test[test["ItemId"].isin(train.ItemId)]
    session_group = test.groupby("SessionId").size()
    session_more_than1 = [key for key, value in session_group.items() if value >= 2]
    test = test[test["SessionId"].isin(session_more_than1)]
    # prepare data for MCRPN models.....
    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId', 'Time'],  inplace = True)
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
        if len(value) > 2:
            for i in range(2, len(value)):
                targets_train.append(value[i])
                features_train.append(value[:i])

        targets_train.append(value[-1])
        features_train.append(value[:-1])
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
        targets_test.append(value[-1])
        features_test.append(value[:-1])
            
    item_no = item_no +1
    return [features_train, targets_train], [features_test, targets_test], item_no


def split_data_temp(data, tes_days = DAYS):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    # Last day data is used  for testing of models.
    session_train = session_max_times[session_max_times < tmax-(86400 *tes_days)].index
    session_test = session_max_times[session_max_times >= tmax-(86400 *tes_days)].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId', 'Time'],  inplace = True)
    return train, test
    


def split_data_rsc15_baseline( data):
    train, test = split_data_temp(data)
    
    #train.to_csv("rsc15_train_full.txt", sep = "\t", index = False)
    #test.to_csv("rsc15_test.txt", sep = "\t", index = False)
    
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    
    
    difference = datetime.fromtimestamp(train.Time.max()) - datetime.fromtimestamp(train.Time.min())
    print("Number of training days:", difference.days)
    
    difference = datetime.fromtimestamp(test.Time.max()) - datetime.fromtimestamp(test.Time.min())
    print("Number of test days:", difference.days)
    
    
    train_validation, test_validation = split_data_temp(train)
    
    #train_validation.to_csv("rsc15_train_tr.txt", sep = "\t", index = False)
    #test_validation.to_csv("rsc15_train_valid.txt", sep = "\t", index = False)
    
    unique_items_ids = data["ItemId"].unique()
    
    session_key = "SessionId"
    item_key = "ItemId"
    index_session = train.columns.get_loc( session_key)
    index_item = train.columns.get_loc( item_key )     
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
   
def split_data_rsc15_knn( data):
    train, test = split_data_temp(data)
    
    train.to_csv("rsc15_train_full.txt", sep = "\t", index = False)
    test.to_csv("rsc15_test.txt", sep = "\t", index = False)
    
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    
    
    difference = datetime.fromtimestamp(train.Time.max()) - datetime.fromtimestamp(train.Time.min())
    print("Number of training days:", difference.days)
    
    difference = datetime.fromtimestamp(test.Time.max()) - datetime.fromtimestamp(test.Time.min())
    print("Number of test days:", difference.days)
    
    
    train_validation, test_validation = split_data_temp(train)
    
    train_validation.to_csv("rsc15_train_tr.txt", sep = "\t", index = False)
    test_validation.to_csv("rsc15_train_valid.txt", sep = "\t", index = False)
    
    unique_items_ids = data["ItemId"].unique()
    
    session_key = "SessionId"
    item_key = "ItemId"
    index_session = train.columns.get_loc( session_key)
    index_item = train.columns.get_loc( item_key )
    
    
    return train, test, unique_items_ids

#
# if __name__ == '__main__':
#     path = "datasets/rsc15/yoochoose-clicks"
    
#     dataset = load_data_rsc15(path)
#     filter_data = filter_data_rsc15(dataset)
#     train, test = split_data_rsc15(filter_data)
#     #features_train, targets_train, features_test, targets_test, item_no = split_data(filter_data)




