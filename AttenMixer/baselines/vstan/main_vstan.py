# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024

@author: shefai
"""
from AttenMixer.digi_preprocessing import *
from AttenMixer.lastfm_preprocessing import *
from AttenMixer.gowalla_preprocessing import *
from AttenMixer.rsc15 import *
from AttenMixer.retail_rocket import *
from AttenMixer.baselines.vstan.vstan  import *
from AttenMixer.accuracy_measures import *
from pathlib import Path
from tqdm import tqdm


class VSTAN_MAIN:
    def __init__(self, data_path, result_path, dataset = "diginetica"):
        self.dataset = dataset
        self.result_path = result_path

        if dataset == "diginetica":
            self.k = 500
            self.sample_size = 500
            self.lambda_spw = 0.98
            self.lambda_snh = 51
            self.lambda_inh = 3.4
            self.lambda_idf = 25
            print(self.dataset)
            name = "train-item-views.csv"
            obj = DIGI()
            train_features, test_features, n_node, train, test, _, _,  _, _ =  obj.data_load(data_path / name)
            self.train_data = train
            self.unique_items_ids  = self.train_data.ItemId.unique()
            #original_ser = list(test.SessionId.unique())[:2000]
            #test = test[  test.SessionId.isin(original_ser) ]
            self.test_data = test
            
            
        elif dataset == 'gowalla':
            
            self.k = 620
            self.sample_size = 2500
            self.lambda_spw = 0.94
            self.lambda_snh = 80
            self.lambda_inh = 3.8
            self.lambda_idf = 15
            print(self.dataset)
            name = "loc-gowalla_totalCheckins.txt.gz"
            obj = Gowalla()
            train_features, test_features, n_node, train, test, _, _,  _, _   = obj.data_load(data_path / name)
            original_ser = list(test.SessionId.unique())[:2000]
            test = test[  test.SessionId.isin(original_ser) ]
            self.train_data = train
            self.unique_items_ids  = self.train_data.ItemId.unique()
            self.test_data = test
            
            
            
        elif dataset == "lastfm":
            
            self.k = 1200
            self.sample_size = 2500
            self.lambda_spw = 0.98
            self.lambda_snh = 51
            self.lambda_inh = 0.51
            self.lambda_idf = 1
            
            print(self.dataset)
            name = "userid-timestamp-artid-artname-traid-traname.tsv"
            obj = LastFm()
            _, _, _, train, test = obj.data_load(data_path / name)
            self.train_data = train
            self.unique_items_ids  = self.train_data.ItemId.unique()
            self.test_data = test

        elif dataset == "retailrocket": 
            
            self.k = 150
            self.sample_size = 190
            self.lambda_spw = 11.206
            self.lambda_snh = 5
            self.lambda_inh = 3
            self.lambda_idf = 1

            name = "events.csv"
            data =  load_data_retail(data_path / name)
            data = filter_data(data)
            train, test = split_data_only(data)

            self.train_data = train
            self.unique_items_ids  = self.train_data.ItemId.unique()
            self.test_data = test  
        
        else:
            print("Mention your datatypes")
            
            
            
    def fit_(self, mrr, hirate):
        
        obj1 = VSKNN_STAN(k = self.k,  sample_size = self.sample_size, lambda_spw = self.lambda_spw, lambda_snh = self.lambda_snh, lambda_inh = self.lambda_inh, lambda_idf = self.lambda_idf )
        obj1.fit(self.train_data)
        
        session_key ='SessionId'
        time_key='Time'
        item_key= 'ItemId'
        
        # Intialize accuracy measures.....
        MRR_dictionary = dict()
        for i in mrr:
            MRR_dictionary["MRR_"+str(i)] = MRR(i)
            
        HR_dictionary = dict()
        for i in hirate:
            HR_dictionary["HR_"+str(i)] = HR(i)
        
        test_data = self.test_data
        test_data.sort_values([session_key, time_key], inplace=True)
        items_to_predict = self.unique_items_ids
        # Previous item id and session id....
        prev_iid, prev_sid = -1, -1
        
        print("Starting predicting")
        for i in tqdm(range(len(test_data))):
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
    
                for key in MRR_dictionary:
                    MRR_dictionary[key].add(preds, iid)
                    
                
                # Calculate the Recall values
                for key in HR_dictionary:
                    HR_dictionary[key].add(preds, iid)
                
            prev_iid = iid
            
        # get the results of MRR values.....
        result_frame = pd.DataFrame()    
        for key in MRR_dictionary:
            print(key +"   "+ str(  MRR_dictionary[key].score()    ))
            result_frame[key] =   [MRR_dictionary[key].score()]
            
            
        # get the results of MRR values.....    
        for key in HR_dictionary:
            print(key +"   "+ str(  HR_dictionary[key].score()    ))
            result_frame[key] = [HR_dictionary[key].score()]
        
        
        name = "AttenMixer_VSTAN_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path/name, sep = "\t", index = False) 
        
       
        
        
        
        
        


