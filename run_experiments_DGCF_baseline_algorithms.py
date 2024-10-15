
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
import traceback, os
from pathlib import Path
import argparse
from topn_baselines_neurals.Data_manager.Gowalla_Yelp_Amazon_DGCF import Gowalla_Yelp_Amazon_DGCF 
def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)
    return recommender_object
if __name__ == '__main__':
    commonFolderName = "experiments_results"
    model = "DGCF"
    dataset_name = "gowalla" # yelp2018, gowalla, amazonbook
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='yelp2018', help="yelp2018, gowalla, amazonbook")

    # python run_experiments_DGCF_baseline_algorithms.py --dataset yelp2018
    # python run_experiments_DGCF_baseline_algorithms.py --dataset gowalla
    # python run_experiments_DGCF_baseline_algorithms.py --dataset amazonbook


    args = parser.parse_args()
    dataset_name = args.dataset
    commonFolderName = "results"
    data_path = Path("data/DGCF/")
    data_path = data_path.resolve()
    validation_set = False
    datasetName = args.dataset+".pkl"
    model = "DGCF"
    validation_set = False
    dataset_object = Gowalla_Yelp_Amazon_DGCF()
    URM_train, URM_test = dataset_object._load_data_from_give_files(validation=validation_set, data_name = data_path / dataset_name)
    ICM_all = None
    UCM_all = None
    saved_results = "/".join([commonFolderName,model,args.dataset] )
    # If directory does not exist, create
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
