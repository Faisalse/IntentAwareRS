#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import zipfile, shutil
from topn_baselines_neurals.Data_manager.DataReader import DataReader
from topn_baselines_neurals.Data_manager.DatasetMapperManager import DatasetMapperManager
from topn_baselines_neurals.Data_manager.Movielens._utils_movielens_parser import _loadURM, _loadICM_genres_years
from topn_baselines_neurals.Data_manager.split_functions.KGIN_given_train_test_splits import split_train_test_validation


class lastFM_AmazonBook_AliBabaFashion_KGIN(DataReader):

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
            with open(zipFile_path / "train.txt") as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        train_list.append(set(items))
            
            with open(zipFile_path / "test.txt") as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [i for i in l[1:]]
                        test_list.append(set(items))

        except FileNotFoundError:
            print(f"File not found: {zipFile_path}")

        URM_dataframe = self.convert_dictionary_to_dataframe_DGCF(train_list, test_list)

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)
        
        if validation == True:
            URM_train, URM_Validation, URM_test = split_train_test_validation(loaded_dataset, test_list, validation=validation)
            return URM_train, URM_Validation, URM_test
        else:
            URM_train, URM_test = split_train_test_validation(loaded_dataset, test_list,   validation=validation)
            return URM_train, URM_test
        
    def convert_dictionary_to_dataframe_DGCF(self, train_list, test_list):

        full_data = dict()
        
        for i in range(len(train_list)):
            
            temp = train_list[i]
            temp.update(test_list[i])
            full_data[i] = temp   
        expanded_data = [(key, value) for key, values in full_data.items() for value in values]
        # Create DataFrame
        URM_dataframe = pd.DataFrame(expanded_data, columns=['UserID', 'ItemID'])
        URM_dataframe["Data"] = 1
        URM_dataframe['UserID']= URM_dataframe['UserID'].astype(str)
        URM_dataframe['ItemID']= URM_dataframe['ItemID'].astype(str)

        return URM_dataframe


       
        











