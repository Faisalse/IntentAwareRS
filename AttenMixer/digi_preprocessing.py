# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:07:25 2024

@author: shefai
"""

import argparse
import time
import csv
import pickle
import operator
import pandas as pd
import numpy as np
import datetime as dt
NUMBER_OF_TESTING_DAYS = 7

class DIGI:
    
    def __init__(self):
        pass
    def data_load(self, path):
        data = pd.read_csv(path, sep=';', header=0, usecols=[0,2,3,4], dtype={0:np.int32, 1:np.int64, 2:np.int32,3:str})
        data = data.iloc[:100000, :]
        #data.columns = ['sessionId', 'TimeStr', 'itemId']
        data['Time'] = data['eventdate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp()) #This is not UTC. It does not really matter.
        del(data['eventdate'])
        del data["timeframe"]
        data.rename(columns = {"sessionId": "SessionId", "itemId": "ItemId"}  , inplace = True)
        
        train, test  = self.data_preprocessing(data)
        train_tr, test_tr= self.data_preprocessing(train)
        print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
        print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
        
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
        index2word = {}
        item_no = 1
    
        for key, values in session_item_train.items():
            length = len(session_item_train[key])
            for i in range(length):
                if session_item_train[key][i] in word2index:
                    session_item_train[key][i] = word2index[session_item_train[key][i]]
                else:
                    word2index[session_item_train[key][i]] = item_no
                    index2word[item_no] = session_item_train[key][i]
                    session_item_train[key][i] = item_no
                    item_no +=1                       
        features = []
        targets = []
        for value in session_item_train.values():
            for i in range(1, len(value)):
                targets.append(value[i])
                features.append(value[:i])
                
                
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
                    index2word[item_no] = session_item_test[key][i]
                    session_item_test[key][i] = item_no
                    item_no +=1
                    
        features1 = []
        targets1 = []
        for value in session_item_test.values():
            for i in range(1, len(value)):
                targets1.append(value[-i])
                features1.append(value[:-i])
        all_train_sequence = []
        for value in session_item_train.values():
                all_train_sequence.append(value)     
        train_features = [features, targets]
        test_features = [features1, targets1]
        return train_features, test_features, item_no, train, test, train_tr, test_tr,  word2index, index2word
    
    
    def data_preprocessing(self, data):
    
        session_lengths = data.groupby('SessionId').size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
        
        item_supports = data.groupby('ItemId').size()
        data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
        
        session_lengths = data.groupby('SessionId').size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
    
        tmax = data.Time.max()
        session_max_times = data.groupby('SessionId').Time.max()
        session_train = session_max_times[session_max_times < tmax-86400 * NUMBER_OF_TESTING_DAYS ].index
        session_test = session_max_times[session_max_times > tmax-86400 * NUMBER_OF_TESTING_DAYS].index
    
    
        train = data[np.in1d(data.SessionId, session_train)]
        trlength = train.groupby('SessionId').size()
        train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
        
        test = data[np.in1d(data.SessionId, session_test)]
        test = test[np.in1d(test.ItemId, train.ItemId)]
        
        tslength = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
        return train, test
    
# obj1 = DIGI()
# dataset = 'diginetica/train-item-views.csv'
# train_features, test_features, item_no, train, test, train_tr, test_tr,  word2index, index2word = obj1.data_load(dataset)
# print("Faisal")


# session_key ='SessionId'
# item_key= 'ItemId'

# index_session = test.columns.get_loc( session_key)
# index_item = test.columns.get_loc( item_key )


# unique_sessions = test.SessionId.unique()

# from tqdm import tqdm
# for sId in tqdm(unique_sessions):
    
#     tempdf = test[test.SessionId == sId]
#     currentSessItems = tempdf.iloc[:, index_item]
    
    
#     currentSessItems = [word2index[i]    for i in currentSessItems]
#     currentSessItems = [ [ currentSessItems[:-1],  currentSessItems[:-1]    ], [  currentSessItems[-1],  currentSessItems[-1] ] ]
    

















# #%%
# train_val.to_csv("lastfm_train_tr.txt", sep="\t", index = False)
# test_val.to_csv("lastfm_train_valid.txt", sep="\t", index = False)