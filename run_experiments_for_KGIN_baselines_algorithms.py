
from topn_baselines_neurals.Recommenders.Knowledge_Graph_based_Intent_Network_KGIN_WWW.run_experiments_KGIN_ import *
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Data_manager.lastFM_AmazonBook_AliBabaFashion_KGIN import lastFM_AmazonBook_AliBabaFashion_KGIN 
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
import traceback, os


from pathlib import Path
import argparse


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)
    return recommender_object
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='lastFm', help="alibabaFashion / amazonBook / lastFm")
    args = parser.parse_args()
    dataset_name = args.dataset
    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    commonFolderName = "results"
    
    task = "training"
    data_path = Path("data/KGIN/"+dataset_name)
    data_path = data_path.resolve()
    # If directory does not exist, create

    ############### prepare baseline data ###############
    baseline_models = "baseline_models"
    validation_set = False
    dataset_object = lastFM_AmazonBook_AliBabaFashion_KGIN()
    URM_train, URM_test = dataset_object._load_data_from_give_files(data_path, validation=validation_set)
    ICM_all = None
    UCM_all = None
    ################# end to prepare baseline data #################
    
    if dataset_name == "lastFm":
        
        dim=64
        lr= 0.0001
        sim_regularity=0.0001
        batch_size=1024
        node_dropout=True
        node_dropout_rate=0.5 
        mess_dropout=True 
        mess_dropout_rate=0.1 
        gpu_id=0
        context_hops=3
        epoch = 60
    elif dataset_name == "alibabaFashion":
        
        dim=64
        lr= 0.0001
        sim_regularity=0.0001
        batch_size=1024 
        node_dropout=True
        node_dropout_rate=0.5 
        mess_dropout=True 
        mess_dropout_rate=0.1 
        gpu_id=0
        context_hops=3
        epoch = 60
    elif dataset_name == "amazonBook":
        dataset= data_path
        dim=64
        lr= 0.0001
        sim_regularity=0.00001
        batch_size=1024 
        node_dropout=True
        node_dropout_rate=0.5 
        mess_dropout=True 
        mess_dropout_rate=0.1 
        gpu_id=0
        context_hops=3
        epoch = 200
    else:
        print("If you do not mention HPs values then it will default HP values of lastFm dataset")
        dim=64,
        lr = 0.0001
        sim_regularity=0.0001
        batch_size=1024 
        node_dropout=True
        node_dropout_rate=0.5 
        mess_dropout=True 
        mess_dropout_rate=0.1 
        gpu_id=0
        context_hops=3
    result_df = run_experiments_KGIN_model(dataset=data_path, dim=dim, lr = lr, sim_regularity=sim_regularity, batch_size=batch_size, 
                                           node_dropout=node_dropout, node_dropout_rate=node_dropout_rate, mess_dropout=mess_dropout, 
                                           mess_dropout_rate=mess_dropout_rate, gpu_id=gpu_id, context_hops=context_hops, epoch = epoch)
    
    result_path = Path()
    saved_results_dl = "/".join([commonFolderName,"KGIN", dataset_name] )
    if not os.path.exists(saved_results_dl):
        os.makedirs(saved_results_dl)
    
    
    result_df.to_csv(saved_results_dl+"KGIN_model_"+dataset_name+".text", index = False, sep = "\t")
    ### experiments for baseline models.....................
    baseline_models = "baseline_models"
    validation_set = False
    dataset_object = lastFM_AmazonBook_AliBabaFashion_KGIN()
    URM_train, URM_test = dataset_object._load_data_from_give_files(data_path, validation=validation_set)
    ICM_all = None
    UCM_all = None
    saved_results = "/".join([commonFolderName,"KGIN", baseline_models, dataset_name] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    output_root_path = saved_results+"/"
    recommender_class_list = [
        Random,
        TopPop,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        EASE_R_Recommender
        ]
    evaluator = EvaluatorHoldout(URM_test, [5,10, 20], exclude_seen=True)
    logFile = open(output_root_path + "result_all_algorithms.txt", "a")
    for recommender_class in recommender_class_list:
        try:
            print("Algorithm: {}".format(recommender_class))
            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15}
            elif(dataset_name == "Music"):
                if isinstance(recommender_object, RP3betaRecommender):
                    fit_params = {"topK": 814, "alpha": 0.13435726416026783, "beta": 0.27678107504384436, "normalize_similarity": True}
                else:
                    fit_params = {}
            elif(dataset_name == "Beauty"):
                print("********************************")
                if isinstance(recommender_object, P3alphaRecommender):
                    fit_params = {"topK": 790, "alpha": 0.0, "normalize_similarity": False}
                
                elif isinstance(recommender_object, RP3betaRecommender):
                    fit_params = {"topK": 1000, "alpha": 0.0, "beta": 0.0, "normalize_similarity": False}
                else:
                    fit_params = {}
            else:
                fit_params = {}
            recommender_object.fit(**fit_params)
            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)
            recommender_object.save_model(output_root_path, file_name = "temp_model.zip")
            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
            recommender_object.load_model(output_root_path, file_name = "temp_model.zip")
            os.remove(output_root_path + "temp_model.zip")
            results_run_2, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)
            if recommender_class not in [Random]:
                assert results_run_1.equals(results_run_2)
            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender_class, results_run_string_1))
            logFile.flush()
        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()

    
    


    


    


    


