# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024

@author: shefai
"""

from HIDE.datasets.process_tmall_class import *
from HIDE.baselines.SR.sr  import SequentialRules
from pathlib import Path
from HIDE.accuracy_measures import *


class SequentialRulesMain:
    
    def __init__(self, data_path, result_path, dataset = "tmall"):
        self.dataset = dataset
        self.result_path = result_path
        if dataset == "lastfm":
            pass
        
        elif dataset == 'tmall':
            
            # sr-steps=20-weighting=same-pruning=20
            self.steps = 20
            self.weighting = "same"
            self.pruning = 20
            self.session_weighting = "div" 
            
            
            name = "dataset15.csv"
            obj1 = Tmall()
            tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)

            tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
            tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
            num_node = len(obj1.item_dict) + 1
            tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
            self.train_data = obj1.train_convert_data_for_baselines( tr_seqs, tr_dates, tr_labs, tr_ids )
            self.unique_items_ids  = self.train_data.ItemId.unique()
            # test sequences....
            te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
            
            self.test_data = obj1.train_convert_data_for_baselines( te_seqs, te_dates, te_labs, te_ids )
                
        else:
            print("Mention your datatypes")
            
            
    def fit_(self, topKList):
        
        obj1 = SequentialRules(steps = self.steps, weighting = self.weighting, pruning = self.pruning, session_weighting = self.session_weighting)
        obj1.fit(self.train_data)
        
        session_key ='SessionId'
        time_key='Time'
        item_key= 'ItemId'
        
        # Intialize accuracy measures.....
        performance_measures = dict()
        for i in topKList:
            performance_measures["MRR_"+str(i)] = MRR(i)
            performance_measures["Pre_"+str(i)] = Precision(i)
            performance_measures["HR_"+str(i)] = HR(i)
            
        test_data = self.test_data
        test_data.sort_values([session_key, time_key], inplace=True)
        items_to_predict = self.unique_items_ids
        # Previous item id and session id....
        prev_iid, prev_sid = -1, -1
        
        print("Starting predicting")
        for i in range(len(test_data)):
            sid = test_data[session_key].values[i]
            iid = test_data[item_key].values[i]
            ts = test_data[time_key].values[i]
            
            if prev_sid != sid:
                # this will be called when there is a change of session....
                prev_sid = sid
            else:
                # prediction starts from here.......
                preds = obj1.predict_next(prev_iid, items_to_predict)
                preds[np.isnan(preds)] = 0
    #             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                preds.sort_values( ascending=False, inplace=True )    
    
    
                for key in performance_measures:
                    performance_measures[key].add(preds, iid)
                
            prev_iid = iid
            
        # get the results of MRR values.....
        result_frame = pd.DataFrame()    
        for key in performance_measures:
            print(key +"   "+ str(  performance_measures[key].score()    ))
            result_frame[key] =   [performance_measures[key].score()]
         
            
        name = "HIDE_SR_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path / name, sep = "\t", index = False) 
        
       
        
        
        
        
        


