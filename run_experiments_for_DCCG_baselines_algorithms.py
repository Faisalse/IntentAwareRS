
from topn_baselines_neurals.Recommenders.DCCG.main import *
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Data_manager.Gowalla_AmazonBook_Tmall_DCCG import Gowalla_AmazonBook_Tmall_DCCG 
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
import traceback, os
from pathlib import Path
import argparse
import pandas as pd
import time


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
    parser.add_argument('--dataset', type = str, default='gowalla', help="amazonbook / gowalla / tmall")
    args = parser.parse_args()
    dataset_name = args.dataset
    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    data_path = Path("data/DCCG/"+dataset_name)
    data_path = data_path.resolve()
    start = time.time()
    
    best_epoch = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=True, epoch = 500)
    print("Start tuning by Best Epoch Value"+str(best_epoch))
    best_score = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=False, epoch = best_epoch)
    
    print("Recall@20: ", str(best_score["recall"][0]), " Recall@40: ", str(best_score["recall"][1]),
                        " NDCG@20: ",str(best_score["ndcg"][0]), " NDCG@40:", str(best_score["ndcg"][1]))
    end = time.time()
    df = pd.DataFrame()
    df["Recall@20"] = [best_score["recall"][0]]
    df["Recall@40"] = [best_score["recall"][1]]
    df["NDCG@20"] = [best_score["ndcg"][0]]
    df["NDCG@40"] = [best_score["ndcg"][1]]
    df["Time(seconds)"] = [end - start]

    commonFolderName = "results"
    model = "DCCG"
    saved_results = "/".join([commonFolderName, model] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    df.to_csv(saved_results + "/"+args.dataset+"_DCCG.txt", index = False)
    


    ############### prepare baseline data ###############
    baseline_models = "baseline_models"
    validation_set = False
    dataset_object = Gowalla_AmazonBook_Tmall_DCCG()
    URM_train, URM_test = dataset_object._load_data_from_give_files(data_path, validation=validation_set)
    ICM_all = None
    UCM_all = None
    
    result_path = Path()
    ### experiments for baseline models.....................
    baseline_models = "baseline_models"
    validation_set = False
    ICM_all = None
    UCM_all = None
    saved_results = "/".join([commonFolderName,"DCCG", baseline_models, dataset_name] )
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
    evaluator = EvaluatorHoldout(URM_test, [20, 40], exclude_seen=True)
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

    
    


    


    


    


