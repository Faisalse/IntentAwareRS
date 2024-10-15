import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd
import numpy as np


# -------------- tmall dataset duration --------
# MIN TIME Wednesday, May 20, 2015 12:00:00 AM GMT+02:00 DST

# MAX TIME Thursday, November 12, 2015 12:00:01 AM GMT+01:00

# DURATION
# 5 months 23 days
# or 25 weeks 1 days
# 176 days


# Consider first 120000 sessions fdo
# TIME MIN: Wednesday, May 20, 2015 12:00:00 AM GMT+02:00 DST
# Thursday, November 12, 2015 12:00:01 AM GMT+01:00

class Tmall:
    
    def __init__(self):
        self.item_dict = {}
        self.TEST_SECONDS = 100
        
    
    def data_load(self, path):
        
        data = pd.read_csv(path, sep = "\t")
        data = data[data["SessionId"] < 120000]


        session_key ='SessionId'
        item_key= 'ItemId'
        time_key='Time'


        index_session = data.columns.get_loc( session_key )
        index_item = data.columns.get_loc( item_key )
        index_time = data.columns.get_loc( time_key )


        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for row in data.itertuples(index=False):
            sessid = int(row[index_session])
            if curdate and not curid == sessid:
                date = curdate
                sess_date[curid] = date
            curid = sessid
            item = int(row[index_item])
            curdate = float(row[index_time])

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        date = float(row[index_time])
        sess_date[curid] = date


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
            if len(filseq) < 2 or len(filseq) > 40:
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

        # the last of 100 seconds for test
        splitdate = maxdate - self.TEST_SECONDS

        print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
        tra_sess = filter(lambda x: x[1] < splitdate, dates)
        tes_sess = filter(lambda x: x[1] > splitdate, dates)

        # Sort sessions by date
        tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        return tra_sess, tes_sess, sess_clicks
    
    
    
    # Convert training sessions to sequences and renumber items to start from 1
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
    
    def process_seqs(self, iseqs, idates):
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
    
    
    
    def train_convert_data_for_baselines(self, tr_seqs, tr_dates, tr_labs, tr_ids):
        
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
    
    
# obj1 = Tmall()
# dataset = 'tmall/dataset15.csv'
# tra_sess, tes_sess, sess_clicks = obj1.data_load(dataset)
# tra_ids, tra_dates, tra_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
# tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)

# tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(tra_seqs, tra_dates)
# te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)

# tra = (tr_seqs, tr_labs)
# tes = (te_seqs, te_labs)

# if not os.path.exists('tmall'):
#     os.makedirs('tmall')
# pickle.dump(tra, open('tmall/train.txt', 'wb'))
# pickle.dump(tes, open('tmall/test.txt', 'wb'))
# pickle.dump(tra_seqs, open('tmall/all_train_seq.txt', 'wb'))

# # convert data to baselines models.....
# from pathlib import Path
# print(len(obj1.item_dict))

# dataframe_train = obj1.train_convert_data_for_baselines( tr_seqs, tr_dates, tr_labs, tr_ids )
# vali_train, vali_test = obj1.split_validation(dataframe_train)


# test = obj1.train_convert_data_for_baselines( te_seqs, te_dates, te_labs, te_ids )

# d1 = Path("tmall/tmall_train_full.txt")
# d2 = Path("tmall/tmall_test.txt")
# d3 = Path("tmall/tmall_train_tr.txt")
# d4 = Path("tmall/tmall_train_valid.txt")


# dataframe_train.to_csv(d1, sep = "\t", index = False)
# test.to_csv(d2, sep = "\t", index = False)
# vali_train.to_csv(d3, sep = "\t", index = False)
# vali_test.to_csv(d4, sep = "\t", index = False)
