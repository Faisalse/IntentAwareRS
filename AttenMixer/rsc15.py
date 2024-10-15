# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:31:24 2024

@author: shefai
"""

import time
import csv
import operator
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path 



class Data_processing:
    
    def __init__(self, dataset = "diginetica", path = "datasets/diginetica/train-item-views.csv"):
        self.dataset = dataset
        self.path = path
        self.item_dict = {}
        
        if dataset in ['yoochoose1_64', 'yoochoose1_4']:
            
            data = pd.read_csv(path)
            data.columns = ['sessionId','timestamp','itemId','category']
            
            # group_sizes = data.groupby('sessionId').size()

            # # Filter out groups with size more than 20
            # filtered_groups = group_sizes[group_sizes <= 19]

            # # Extract rows from the original DataFrame based on the filtered groups
            # filtered_df = data[data['sessionId'].isin(filtered_groups.index)]
            data.to_csv(path, sep = ",", index = False)
            
        
    
    def data_load(self):
        
        print("-- Starting @ %ss" % datetime.datetime.now())
        with open(self.path, "r") as f:
            if self.dataset in ['yoochoose1_64', 'yoochoose1_4']:
                reader = csv.DictReader(f, delimiter=',')
            else:
                reader = csv.DictReader(f, delimiter=';')
            sess_clicks = {}
            sess_date = {}
            ctr = 0
            curid = -1
            curdate = None
            for data in tqdm(reader):
                sessid = data['sessionId']
                if curdate and not curid == sessid:
                    date = ''
                    if self.dataset in ['yoochoose1_64', 'yoochoose1_4']:
                        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
                    else:
                        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                    sess_date[curid] = date
                curid = sessid
                if self.dataset in ['yoochoose1_64', 'yoochoose1_4']:
                    item = data['itemId']
                else:
                    item = data['itemId'], int(data['timeframe'])
                curdate = ''
                if self.dataset in ['yoochoose1_64', 'yoochoose1_4']:
                    curdate = data['timestamp']
                else:
                    curdate = data['eventdate']

                if sessid in sess_clicks:
                    sess_clicks[sessid] += [item]
                else:
                    sess_clicks[sessid] = [item]
                ctr += 1
            date = ''
            if self.dataset in ['yoochoose1_64', 'yoochoose1_4']:
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                for i in list(sess_clicks):
                    sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                    sess_clicks[i] = [c[0] for c in sorted_clicks]
            sess_date[curid] = date
        print("-- Reading data @ %ss" % datetime.datetime.now())

        # Filter out length 1 sessions
        for s in list(sess_clicks):
            if len(sess_clicks[s]) == 1:
                del sess_clicks[s]
                del sess_date[s]

        # Count number of times each item appears
        iid_counts = {}
        for s in sess_clicks:
            seq = sess_clicks[s]
            for iid in seq:
                if iid in iid_counts:
                    iid_counts[iid] += 1
                else:
                    iid_counts[iid] = 1

        sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

        length = len(sess_clicks)
        for s in list(sess_clicks):
            curseq = sess_clicks[s]
            filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
            if len(filseq) < 2:
                del sess_clicks[s]
                del sess_date[s]
            else:
                sess_clicks[s] = filseq

        # Split out test set based on dates
        dates = list(sess_date.items())
        maxdate = dates[0][1]

        for _, date in dates:
            if maxdate < date:
                maxdate = date

        # 7 days for test
        splitdate = 0
        if self.dataset in ['yoochoose1_64', 'yoochoose1_4']:
            splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
        else:
            splitdate = maxdate - 86400 * 7

        print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
        tra_sess = filter(lambda x: x[1] < splitdate, dates)
        tes_sess = filter(lambda x: x[1] > splitdate, dates)

        # Sort sessions by date
        tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
        tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
        
        return tra_sess, tes_sess, sess_clicks
    
    
    def obtian_tra(self, tra_sess, sess_clicks):
        train_ids = []
        train_seqs = []
        train_dates = []
        item_ctr = 1
        for s, date in tra_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in self.item_dict:
                    outseq += [self.item_dict[i]]
                else:
                    outseq += [item_ctr]
                    self.item_dict[i] = item_ctr
                    item_ctr += 1
            if len(outseq) < 2:  # Doesn't occur
                continue
            train_ids += [s]
            train_dates += [date]
            train_seqs += [outseq]

        return train_ids, train_dates, train_seqs
    


    def obtian_tes(self, tes_sess, sess_clicks):
        test_ids = []
        test_seqs = []
        test_dates = []
        for s, date in tes_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in self.item_dict:
                    outseq += [self.item_dict[i]]
            if len(outseq) < 2:
                continue
            test_ids += [s]
            test_dates += [date]
            test_seqs += [outseq]
        return test_ids, test_dates, test_seqs
    
    
    
    def process_seqs_train(self, iseqs, idates):
        out_seqs = []
        out_dates = []
        labs = []
        ids = []
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]
                
                
                
        if self.dataset ==  "yoochoose1_4":
            print("yoochoose ratio:    1/4")
            split4  = int(len(out_seqs) / 4)
            
            out_seqs = out_seqs[-split4:]
            out_dates = out_dates[-split4:]
            labs = labs[-split4:]
            ids = ids[-split4:]
            
            
        
        if self.dataset ==  "yoochoose1_64":
            print("yoochoose ratio:    1/64")
            split64 = int(len(out_seqs) / 64)
            out_seqs = out_seqs[-split64:]
            out_dates = out_dates[-split64:]
            labs = labs[-split64:]
            ids = ids[-split64:]
            
            
            
                
        return out_seqs, out_dates, labs, ids
    
    
    def process_seqs_test(self, iseqs, idates):
        out_seqs = []
        out_dates = []
        labs = []
        ids = []
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]
                
                
        return out_seqs, out_dates, labs, ids
    
    
    
    def convert_data_for_baselines(self, tr_seqs, tr_dates, tr_labs, tr_ids):
        
        train_temp = []
        time_temp = []
        session_temp = []
        
        for i in range(len(tr_seqs)):
            train_temp.append(tr_seqs[i] + [ tr_labs[i]  ])
            
            # time matching
            t1 = [tr_dates[i] for j in range(len(train_temp[i])) ]
            time_temp.append(t1)
            
            
            # session matching
            t1 = [tr_ids[i] for j in range(len(train_temp[i])) ]
            session_temp.append(t1)
        
        
    
        dataframe = pd.DataFrame()
        
        dataframe["ItemId"] = [element for innerList in train_temp for element in innerList]
        dataframe["SessionId"] = [element for innerList in session_temp for element in innerList]
        dataframe["Time"] = [element for innerList in time_temp for element in innerList]
    
        return dataframe
    
    
    def split_validation(self, train_set, valid_portion = 0.1):
        
        train_set.sort_values(["SessionId", "Time"], inplace=True)
        n_train = int(np.round(len(train_set) * (1. - valid_portion)))
        
        tr_train = train_set.iloc[:n_train, :]
        val_test = train_set.iloc[n_train:, :]
        

        return tr_train, val_test
    
#%%   
# path1 = Path("yoochoose/yoochoose-clicks.dat")
# path2 = Path("diginetica/train-item-views.csv")
 
# obj1 = Data_processing(dataset = "yoochoose1_64", path = path1)
# tra_sess, tes_sess, sess_clicks = obj1.data_load()
# tra_ids, tra_dates, tra_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
# tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)


# tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs_train(tra_seqs, tra_dates)
# te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs_test(tes_seqs, tes_dates)


#%%
# dataframe_train = obj1.convert_data_for_baselines( tr_seqs, tr_dates, tr_labs, tr_ids )
# vali_train, vali_test = obj1.split_validation(dataframe_train)


# test = obj1.convert_data_for_baselines( te_seqs, te_dates, te_labs, te_ids )

# if obj1.dataset in ['yoochoose1_64', 'yoochoose1_4']:
#     d1 = Path("yoochoose/rec15_train_full.txt")
#     d2 = Path("yoochoose/rec15_test.txt")
#     d3 = Path("yoochoose/rec15_train_tr.txt")
#     d4 = Path("yoochoose/rec15_train_valid.txt")
# else:
    
#     d1 = Path("diginetica/diginetica_train_full.txt")
#     d2 = Path("diginetica/diginetica_test.txt")
#     d3 = Path("diginetica/diginetica_train_tr.txt")
#     d4 = Path("diginetica/diginetica_train_valid.txt")


# dataframe_train.to_csv(d1, sep = "\t", index = False)
# test.to_csv(d2, sep = "\t", index = False)
# vali_train.to_csv(d3, sep = "\t", index = False)
# vali_test.to_csv(d4, sep = "\t", index = False)








