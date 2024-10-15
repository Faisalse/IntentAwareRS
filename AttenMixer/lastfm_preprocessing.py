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

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import Counter
SESSION_LENGTH = 8 * 60 * 60 # 8 hours


class LastFm:
    
    def __init__(self):
        pass
        
    
    def data_load(self, path):
        
        data = pd.read_csv(path, sep = "\t", error_bad_lines=False)
        data.columns = ['userid', 'datetime', 'artid', 'artname', 'traid', 'traname']
        del data["artname"]
        del data["traid"]
        del data["traname"]
        #data = data.iloc[:100000, :]
        Ist = list(data.artid)
        
        counter = Counter(Ist)
        # most common 40000 popular tracks.....
        most_common = counter.most_common(40000)
        most_ = [value for value, count in most_common]
        data = data[ data["artid"].isin(most_) ]
        data['TimeTmp'] = pd.to_datetime(data.datetime, format='%Y-%m-%d %H:%M:%S')
        # Sort by TimeTmp
        data.sort_values(by=['userid', 'TimeTmp'], inplace=True)
        users = data.groupby('userid')
        
        
        sessionList = list()
        new_session = 0
        # new group means new user
        for name, group in users:
            time_difference_seconds = ( ( group["TimeTmp"] - group["TimeTmp"].shift(1) ).dt.total_seconds()).fillna(0)
            temp_checking = group["TimeTmp"] - group["TimeTmp"].shift(1)
            summ = 0
            for value in time_difference_seconds:
                summ+=value
                if( summ > SESSION_LENGTH):
                    summ = 0
                    new_session+=1
                sessionList.append(new_session)
            new_session+=1   
                
        data["SessionId"] =   sessionList
        data['TimeTmp'] = data['TimeTmp'].astype(str)
        data['Time'] = data["TimeTmp"].apply(lambda x: datetime.strptime(x.split("+")[0], '%Y-%m-%d %H:%M:%S').timestamp())
        data.rename(columns = {"artid": "ItemId"}, inplace=True)
        del data["userid"]
        del data["datetime"]
        del data["TimeTmp"]
        train, test = self.data_preprocessing(data)
        train_tr, test_tr = self.data_preprocessing(train)
        
        
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
                targets.append(value[-i])
                features.append(value[:-i])
                
                
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
        return train_features, test_features, item_no, train, test, train_tr, test_tr
    def data_preprocessing(self, data):
    
        session_support = data.groupby('SessionId').size()
        session_support = [key for key, value in session_support.items() if 1 < value <=50]
        data = data[data["SessionId"].isin(session_support)  ]
    
        item_supports = data.groupby('ItemId').size()
        item_supports = [key for key, value in item_supports.items() if value > 5]
        data = data[data["ItemId"].isin(item_supports)  ]
                
    
        session_support = data.groupby('SessionId').size()
        session_support = [key for key, value in session_support.items() if value > 1]
        data = data[data["SessionId"].isin(session_support)  ]
    
        train_part = int(len(data) * 0.80)
        train = data.iloc[:train_part, :]
        test = data.iloc[train_part:, :]
    
    
        session_support = train.groupby('SessionId').size()
        session_support = [key for key, value in session_support.items() if value > 1]
        train = train[train["SessionId"].isin(session_support)  ]
    
        test = test[test.ItemId.isin(train.ItemId)]
    
        session_support = test.groupby('SessionId').size()
        session_support = [key for key, value in session_support.items() if value > 1]
        test = test[test["SessionId"].isin(session_support)  ]


        return train, test
    
# obj1 = LastFm()
# dataset = 'userid-timestamp-artid-artname-traid-traname.tsv'
# train_features, test_features, n_nodes, train, test, train_val, test_val = obj1.data_load(dataset)
#validation_tr, validation_te = obj1.validation_spliting(ori_train_data)