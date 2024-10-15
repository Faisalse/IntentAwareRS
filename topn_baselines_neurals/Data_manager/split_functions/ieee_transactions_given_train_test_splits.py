#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from topn_baselines_neurals.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
def split_train_test_validation(loaded_dataset, given_test_data, validation = False):
    """
    The function splits an URM in two matrices selecting the k_out interactions one user at a time
    :param URM:
    :param k_out:
    :param use_validation_set:
    :param leave_random_out:
    :return:
    """
    k_out = 1 
    use_validation_set = validation 
    URM = loaded_dataset.AVAILABLE_URM['URM_all']
    
    updated_test_data = update_item_ids_of_original_data(loaded_dataset.item_original_ID_to_index, given_test_data)


    assert k_out > 0, "k_out must be a value greater than 0, provided was '{}'".format(k_out)

    URM = sps.csr_matrix(URM)
    n_users, n_items = URM.shape

    URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    if use_validation_set:
         URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                                          auto_create_col_mapper=False, n_cols = n_items)
    
    for user_id in range(n_users):

        test_records_items = np.array(list(updated_test_data[user_id]))
        test_data = np.ones(len(test_records_items))

        start_user_position = URM.indptr[user_id]
        end_user_position = URM.indptr[user_id+1]

        user_profile = URM.indices[start_user_position:end_user_position]


        # remove test items from the traning data
        user_profile = list(user_profile)
        for item_ in test_records_items:
            user_profile.remove(item_)
        
        
        user_interaction_items = np.array(user_profile)
        user_interaction_data = np.ones(len(user_profile))


        #Test interactions
        user_interaction_items_test = test_records_items
        user_interaction_data_test = test_data
        URM_test_builder.add_data_lists([user_id]*len(user_interaction_items_test), user_interaction_items_test, user_interaction_data_test)
        if len(user_profile) > 0:
            
        #validation interactions
            if use_validation_set:
                user_interaction_items_validation = user_interaction_items[0:k_out]
                user_interaction_data_validation = user_interaction_data[0:k_out]
                URM_validation_builder.add_data_lists([user_id]*k_out, user_interaction_items_validation, user_interaction_data_validation)


                #Train interactions
                user_interaction_items_train = user_interaction_items[k_out:]
                user_interaction_data_train = user_interaction_data[k_out:]
                URM_train_builder.add_data_lists([user_id]*len(user_interaction_items_train), user_interaction_items_train, user_interaction_data_train)
            else:
                user_interaction_items_train = user_interaction_items
                user_interaction_data_train = user_interaction_data
                URM_train_builder.add_data_lists([user_id]*len(user_interaction_items_train), user_interaction_items_train, user_interaction_data_train)



    URM_train = URM_train_builder.get_SparseMatrix()
    URM_test = URM_test_builder.get_SparseMatrix()
    

    URM_train = sps.csr_matrix(URM_train)
    user_no_item_train = np.sum(np.ediff1d(URM_train.indptr) == 0)

    if user_no_item_train != 0:
        print("Warning: {} ({:.2f} %) of {} users have no Train items".format(user_no_item_train, user_no_item_train/n_users*100, n_users))



    if use_validation_set:
        URM_validation = URM_validation_builder.get_SparseMatrix()

        URM_validation = sps.csr_matrix(URM_validation)
        user_no_item_validation = np.sum(np.ediff1d(URM_validation.indptr) == 0)

        if user_no_item_validation != 0:
            print("Warning: {} ({:.2f} %) of {} users have no Validation items".format(user_no_item_validation, user_no_item_validation/n_users*100, n_users))
        return URM_train, URM_validation, URM_test
    return URM_train, URM_test


def update_item_ids_of_original_data(dictionary_item_original_to_index, list_test_data):
    updated_original_test_data = list()

    for items_set in list_test_data:
        temp = set()
        for item in items_set:
            temp.add(dictionary_item_original_to_index[str(item)])
        
        updated_original_test_data.append(temp)

    return updated_original_test_data

