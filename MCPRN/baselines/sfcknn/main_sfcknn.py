# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024

@author: shefai
"""
# import data preprocessing files....

from MCPRN.baselines.sfcknn.sfcknn  import *
from MCPRN.accuracy_measures import *
from MCPRN.rsc15_data_preprocessing import *
from MCPRN.baselines.vstan.vstan  import *


class SFCKNN_MAIN:
    
    def __init__(self, data_path, result_path, dataset = "yoochoose"):
        self.dataset = dataset
        self.result_path = result_path
        if dataset == "yoochoose":
            
            self.k = 1000
            self.sample_size = 2000
            self.similarity = "cosine"
        
            name = "yoochoose-buys.dat"
            dataset = load_data_rsc15(data_path / name)
            filter_data_ = filter_data_rsc15(dataset)
            _, _, _, self.train_data, self.test_data = split_data_rsc15(filter_data_)
            self.unique_items_ids  = self.train_data.ItemId.unique()     
        else:
            print("Mention your datatypes")
            
            
    def fit_(self, topK):
        
        obj1 = SeqFilterContextKNN(k = self.k,  sample_size = self.sample_size,  similarity = self.similarity)
        obj1.fit(self.train_data)
        session_key ='SessionId'
        time_key='Time'
        item_key= 'ItemId'
        
        # Intialize accuracy measures.....
        performance_measures = dict()
        for i in topK:
            performance_measures["MRR_"+str(i)] = MRR(i)
            performance_measures["NDCG_"+str(i)] = NDCG(i)
            
        
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
            
        # Intialize accuracy measures.....
        name = "MCRPN_SFCKNN_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path /  name, sep = "\t", index = False) 
        
       
        
        
        
        
        


