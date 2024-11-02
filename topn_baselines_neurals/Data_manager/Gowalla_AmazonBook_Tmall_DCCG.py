#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""
import pickle
import pandas as pd
import zipfile, shutil
from topn_baselines_neurals.Data_manager.DataReader import DataReader
from topn_baselines_neurals.Data_manager.DatasetMapperManager import DatasetMapperManager
from topn_baselines_neurals.Data_manager.Movielens._utils_movielens_parser import _loadURM, _loadICM_genres_years
from topn_baselines_neurals.Data_manager.split_functions.DCCG_given_train_test_splits import split_train_test_validation


class Gowalla_AmazonBook_Tmall_DCCG(DataReader):

    DATASET_URL = "https://github.com/NLPWM-WHU/IDS4NR/blob/main/movielens_100k/movielens100k_longtail_data.pkl"
    DATASET_SUBFOLDER = "Movielens100M_given/"
    CONFERENCE_JOURNAL = "KGIN/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_genres"]
    AVAILABLE_UCM = ["UCM_all"]
    IS_IMPLICIT = False
    
    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER
    def _load_data_from_give_files(self, datapath, validation = False ):
        
        
        zipFile_path = datapath
        train_list = list()
        test_list = list()
        try:

            train_file = zipFile_path / 'train.pkl'
            test_file = zipFile_path / 'test.pkl'

            with open(train_file, 'rb') as f:
                train_mat = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_mat = pickle.load(f)
            

        except FileNotFoundError:
            print(f"File not found: {zipFile_path}")


        R = train_mat.todok() # dok --> dictionary of keys....
        train_items, test_set = {}, {} # list of items each user interacted in training data and list of items each user interacted in test data......
        train_uid, train_iid = train_mat.row, train_mat.col
        for i in range(len(train_uid)):
            uid = train_uid[i]
            iid = train_iid[i]
            if uid not in train_items:
                train_items[uid] = [iid]
            else:
                train_items[uid].append(iid)

        test_uid, test_iid = test_mat.row, test_mat.col
        for i in range(len(test_uid)):
            uid = test_uid[i]
            iid = test_iid[i]
            if uid not in test_set:
                test_set[uid] = [iid]
            else:
                test_set[uid].append(iid)

        train_dictionary = train_items
        test_dictionary = test_set

        URM_dataframe = self.convert_dictionary_to_dataframe_DGCF(train_dictionary.copy(), test_dictionary.copy())

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)
        
        if validation == True:
            URM_train, URM_Validation, URM_test = split_train_test_validation(loaded_dataset, test_dictionary, validation=validation)
            return URM_train, URM_Validation, URM_test
        else:
            URM_train, URM_test = split_train_test_validation(loaded_dataset, test_dictionary,   validation=validation)
            return URM_train, URM_test
        
    def convert_dictionary_to_dataframe_DGCF(self, train_dictionary, test_dictionary):

        
        for key, _ in test_dictionary.items():
            train_dictionary[key]+=test_dictionary[key] 
        expanded_data = [(key, value) for key, values in train_dictionary.items() for value in values]
        # Create DataFrame
        URM_dataframe = pd.DataFrame(expanded_data, columns=['UserID', 'ItemID'])
        URM_dataframe["Data"] = 1
        URM_dataframe['UserID']= URM_dataframe['UserID'].astype(str)
        URM_dataframe['ItemID']= URM_dataframe['ItemID'].astype(str)

        return URM_dataframe


       
        











