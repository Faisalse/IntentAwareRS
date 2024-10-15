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
from tqdm import tqdm
from AttenMixer.baselines.stan.stan  import *
from pathlib import Path
from AttenMixer.accuracy_measures import *


class STAN_MAIN:
    
    # self, k, sample_size=5000, sampling='recent', remind=True, 
    #extend=False, lambda_spw=1.02, lambda_snh=5, lambda_inh=2.05 , 
    #session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):
    
    def __init__(self, data_path, result_path,  dataset = "diginetica"):
        self.dataset = dataset
        self.result_path = result_path
        if dataset == "diginetica":
            self.k = 630
            self.sample_size = 2550
            self.lambda_spw = 0.13
            self.lambda_snh = 530
            self.lambda_inh = 0.58
            print(self.dataset)
            name = "train-item-views.csv"
            obj = DIGI()
            train_features, test_features, n_node, original_train, original_test, train_validation, test_validation,  word2index, index2word=  obj.data_load(data_path / name)
            self.train_data = original_train
            self.unique_items_ids  = self.train_data.ItemId.unique()
            # Due to very low prediction capability of Atten_model, we check it performance for first 2000 sessions....
            #original_ser = list(original_test.SessionId.unique())[:2000]
            #original_test = original_test[  original_test.SessionId.isin(original_ser) ]
            self.test_data = original_test
            
        elif dataset == 'gowalla':

            self.k = 510
            self.sample_size = 2550
            self.lambda_spw = 0.16
            self.lambda_snh = 470
            self.lambda_inh = 0.54
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
            
            
            self.k = 750
            self.sample_size = 2800
            self.lambda_spw = 0.17
            self.lambda_snh = 520
            self.lambda_inh = 0.51
            
            print(self.dataset)
            path = "userid-timestamp-artid-artname-traid-traname.tsv"
            obj = LastFm()
            _, _, _, train, test,_,_ = obj.data_load(data_path / name)
            self.train_data = train
            self.unique_items_ids  = self.train_data.ItemId.unique()
            self.test_data = test
            
        else:
            print("Mention your datatypes")
            
            
    def fit_(self, mrr, hitrate):
        
        obj1 = STAN(k = self.k,  sample_size = self.sample_size, lambda_spw = self.lambda_spw, lambda_snh = self.lambda_snh, lambda_inh = self.lambda_inh )
        obj1.fit(self.train_data)
        
        session_key ='SessionId'
        time_key='Time'
        item_key= 'ItemId'
        
        # Intialize accuracy measures.....
        MRR_dictionary = dict()
        for i in mrr:
            MRR_dictionary["MRR_"+str(i)] = MRR(i)
            
        HR_dictionary = dict()
        for i in hitrate:
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
                # Calculate the recall values
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
        # Intialize accuracy measures.....
        name = "AttenMixer_STAN_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path/name, sep = "\t", index = False) 
        
       
        
        
        
        
        


