# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024

@author: shefai
"""

from HIDE.datasets.process_tmall_class import *


from HIDE.baselines.stan.stan  import *
from pathlib import Path
from HIDE.accuracy_measures import *


class STAN_MAIN:
    
    # self, k, sample_size=5000, sampling='recent', remind=True, 
    #extend=False, lambda_spw=1.02, lambda_snh=5, lambda_inh=2.05 , 
    #session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
    
    def __init__(self, data_path, result_path, dataset = "tmall"):
        self.dataset = dataset
        self.result_path = result_path
        if dataset == "lastfm":
            pass
            
            
        elif dataset == 'tmall':
            # stan-k=570-sample_size=2500-lambda_spw=0.17-lambda_snh=520-lambda_inh=0.5
            self.k = 570
            self.sample_size = 2500
            self.lambda_spw = 0.17
            self.lambda_snh = 520
            self.lambda_inh = 0.5
            
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
        
        obj1 = STAN(k = self.k,  sample_size = self.sample_size, lambda_spw = self.lambda_spw, lambda_snh = self.lambda_snh, lambda_inh = self.lambda_inh )
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
                preds = obj1.predict_next(sid, prev_iid, items_to_predict, ts)
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
         
        name = "HIDE_STAN_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path / name, sep = "\t", index = False) 
        
       
        
        
        
        
        


