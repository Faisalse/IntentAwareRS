# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:27:19 2024

@author: shefai
"""

import pandas as pd
import time
import csv
import datetime
import numpy as np


class data_cleaning_DIGI:

    def __init__(self, file):
        data = pd.read_csv(file, sep =";")
        #n = int(len(data) / 3)
        #data = data.iloc[:n,:]
        del data["userId"]
        del data['timeframe']
        data['Time'] = data['eventdate'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').timestamp())
        del data['eventdate']
        data.rename(columns = {"sessionId":"SessionId", 'itemId':'ItemId'}, inplace = True)    
        # remove sessions of length one and items that appear less than five times.....
        session_thresh_hold = 2
        items_thresh_hold = 5

        while True:
            print(data.shape)
            data = data.groupby('SessionId').filter(lambda x: len(x) >= session_thresh_hold)
            data = data.groupby('ItemId').filter(lambda x: len(x) >= items_thresh_hold)

            min_session_lenth = min(data.groupby("SessionId").size())
            min_items = min(data.groupby("ItemId").size())
            if min_session_lenth >= session_thresh_hold and min_items >= items_thresh_hold:
                break


        tmax = data.Time.max()
        session_max_times = data.groupby('SessionId').Time.max()
        session_train = session_max_times[session_max_times < tmax - (86400 * 7)].index
        session_test = session_max_times[session_max_times >= tmax - (86400 * 7)].index
            
        train = data[np.in1d(data.SessionId, session_train)]
        test = data[np.in1d(data.SessionId, session_test)]
        test = test[np.in1d(test.ItemId, train.ItemId)]
        tslength = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
        

        train_seq = train.groupby('SessionId')['ItemId'].apply(list).to_dict()
        
        word2index ={}
        index2word = {}
        item_no = 1
        
        for key, values in train_seq.items():
            length = len(train_seq[key])
            for i in range(length):
                if train_seq[key][i] in word2index:
                    train_seq[key][i] = word2index[train_seq[key][i]]
                    
                else:
                    word2index[train_seq[key][i]] = item_no
                    index2word[item_no] = train_seq[key][i]
                    train_seq[key][i] = item_no
                    item_no +=1

        self.train_seq_f = list()
        self.train_label = list()
        for key, seq_ in train_seq.items():
            self.train_seq_f.append(seq_[:-1])
            self.train_label.append(seq_[-1])

        # test data
        test_seq = test.groupby('SessionId')['ItemId'].apply(list).to_dict()
        
        for key, values in test_seq.items():
            length = len(test_seq[key])
            for i in range(length):
                if test_seq[key][i] in word2index:
                    test_seq[key][i] = word2index[test_seq[key][i]]

        self.test_seq_f = list()
        self.test_label = list()
        for key, seq_ in test_seq.items():
            self.test_seq_f.append(seq_[:-1])
            self.test_label.append(seq_[-1])

        self.complete_test_sequence = test_seq
        self.word2index = word2index
        print("Number of training sessions:   ", len(self.train_seq_f))
        print("Number of test sessions:   ", len(self.test_seq_f))
        print("Number of items:   ", len(word2index))

        
























        
        
