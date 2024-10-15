
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from topn_baselines_neurals.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Data_manager.Movielens.Movielens100MReaderGiven import Movielens100MReaderGiven
import traceback, os
from topn_baselines_neurals.Recommenders.IDS4NR.IDSNR import *
import argparse
from pathlib import Path
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
    parser.add_argument('--dataset', type = str, default='MovieLens', help="MovieLens/Music/Beauty")
    parser.add_argument('--model', type = str, default='LFM', help="LFM or NCF")
    args = parser.parse_args()
    dataset_name = args.dataset

    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model NCF

    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model LFM
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model LFM
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model LFM

    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    commonFolderName = "results"
    data_path = Path("data/ID4SNR/")
    data_path = data_path.resolve()
    validation_set = False
    datasetName = args.dataset+".pkl"
    dataset_object = Movielens100MReaderGiven()
    URM_train, URM_test, UCM_all = dataset_object._load_data_from_give_files(validation=validation_set, data_path = data_path / datasetName)
    ICM_all = None
    resultFolder = "results"
    saved_results = "/".join([resultFolder,"ID4SNR",dataset_name] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    output_root_path = saved_results+"/"

    # Run experiments for IDS4NR
    obj1 = Run_experiments_for_IDSNR(model = args.model, dataset = data_path / datasetName)
    accuracy_measure = obj1.accuracy_values
    accuracy_measure.to_csv(output_root_path+args.dataset+"__"+args.model+".txt", index = False, sep = "\t")

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
