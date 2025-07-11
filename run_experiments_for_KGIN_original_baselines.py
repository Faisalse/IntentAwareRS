from topn_baselines_neurals.Recommenders.Knowledge_Graph_based_Intent_Network_KGIN_WWW.run_experiments_KGIN_ import *
from topn_baselines_neurals.Data_manager.lastFM_AmazonBook_AliBabaFashion_KGIN import lastFM_AmazonBook_AliBabaFashion_KGIN 
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from pathlib import Path
import traceback, os
import argparse
import time

def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):
    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)
    return recommender_object

def parse_args():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazonBook", help="Choose a dataset:[lastFm, alibabaFashion, amazonBook]")
    parser.add_argument('--resolveDataLeakage', nargs='?', default="no", help='yes/no')
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()



if __name__ == '__main__':

    args = parse_args() 
    dataset_name = args.dataset
    resolveLastFMDataLeakageIssue = False

    if args.resolveDataLeakage == "yes":
        resolveLastFMDataLeakageIssue = True
    
    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    data_path = Path("data/KGIN/"+dataset_name)
    data_path = data_path.resolve()
    commonFolderName = "results"
    model = "KGIN"
    saved_results = "/".join([commonFolderName, model] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    start = time.time()

    # optimal hyperparameter values provided by original authors for each dataset....
    if dataset_name == "lastFm":
        args.data_path = data_path
        args.dim=64
        args.lr= 0.0001
        args.sim_regularity=0.0001
        args.batch_size=1024
        args.node_dropout=True
        args.node_dropout_rate=0.5 
        args.mess_dropout=True 
        args.mess_dropout_rate=0.1
        args.context_hops=3
        args.epoch = 509  #### epoch value is taken from the provided training logs.....


    elif dataset_name == "alibabaFashion":
        args.data_path = data_path
        dim=64
        lr= 0.0001
        sim_regularity=0.0001
        batch_size=1024 
        node_dropout=True
        node_dropout_rate=0.5 
        mess_dropout=True 
        mess_dropout_rate=0.1 
        context_hops= 3
        epoch = 209 #epoch value is taken from the provided training logs.....
    
    elif dataset_name == "amazonBook":
        dataset= data_path
        args.data_path = data_path
        dim=64
        lr= 0.0001
        sim_regularity=0.00001
        batch_size=1024 
        node_dropout=True
        node_dropout_rate=0.5 
        mess_dropout=True 
        mess_dropout_rate=0.1
        context_hops=3
        epoch = 579 # epoch value is taken from the provided training logs.....
    else:
        pass
    ############### BASELINE MODELS DATA PREPARATION ###############
    
    validation_set = False
    dataset_object = lastFM_AmazonBook_AliBabaFashion_KGIN()
    URM_train, URM_test = dataset_object._load_data_from_give_files(data_path, dataset = args.dataset, dataLeakage = resolveLastFMDataLeakageIssue, validation=validation_set)
    ICM_all = None
    UCM_all = None

    total_elements = URM_train.shape[0] * URM_train.shape[1]
    non_zero_elements = URM_train.nnz + URM_test.nnz
    density = non_zero_elements / total_elements
    print("Number of users: %s, Items: %d, Interactions: %d, Density %.5f, Number of users with no test items: %d." % 
          (URM_train.shape[0], URM_train.shape[1], non_zero_elements, density, np.sum(np.diff(URM_test.indptr) == 0)))
    ############### END #############################################

    
    ############### RUN EXPERIMENT KGIN MODEL ###############
    result_df = run_experiments_KGIN_model(args, lastFMDataLeakage = resolveLastFMDataLeakageIssue)
    if resolveLastFMDataLeakageIssue == True and args.dataset == "lastFM":
        result_df.to_csv(saved_results+"/"+"KGIN_resolveLastFMDataLeakageIssue_"+dataset_name+".text", index = False, sep = "\t")
    else:
        result_df.to_csv(saved_results+"/"+"KGIN_"+dataset_name+".text", index = False, sep = "\t")
    
    ############### RUN EXPERIMENTS FOR BASELINE MODELS ###############
    
    recommender_class_list = [

        Random,
        TopPop,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        EASE_R_Recommender

        ]
    if args.dataset == "alibabaFashion": # get optimal values.........
        itemkNN_best_HP  = {"topK": 508, "similarity": "cosine", "shrink": 1000, "normalize": True}
        userkNN_best_HP  = {"topK": 146, "similarity": "cosine"}
        RP3alpha_best_HP = {"topK": 777, "alpha": 1.087096950563704, "normalize_similarity": False}
        RP3beta_best_HP  = {"topK": 777, "alpha": 0.5663562161452378, "beta": 0.001085447926739258, "normalize_similarity": True}
        
    elif args.dataset == "lastFm":  # get optimal values.........
        itemkNN_best_HP = {"topK": 144, "similarity": "cosine", "shrink": 1000, "normalize": True}
        userkNN_best_HP = {"topK": 144, "similarity": "cosine", "shrink": 1000, "normalize": True}
        RP3alpha_best_HP = {"topK": 496, "alpha": 0.7681732734954694, "normalize_similarity": False}
        RP3beta_best_HP = {"topK": 350, "alpha": 0.7681732734954694, "beta": 0.4181395996963926, "normalize_similarity": True}
        
    elif args.dataset == "amazonBook":
        itemkNN_best_HP = {"topK": 125, "similarity": "cosine", "shrink": 1000, "normalize": True}
        userkNN_best_HP = {"topK": 454, "similarity": "cosine", "shrink": 1000, "normalize": True}
        RP3alpha_best_HP = {"topK": 496, "alpha": 0.41477903655656115, "normalize_similarity": False}
        RP3beta_best_HP = {"topK": 496, "alpha": 0.44477903655656115, "beta": 0.5968193614337285, "normalize_similarity": True}
    
    evaluator = EvaluatorHoldout(URM_test, [1, 5, 10, 20, 40, 50, 100], exclude_seen=True)
    for recommender_class in recommender_class_list:
        try:
            print("Algorithm: {}".format(recommender_class))
            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15}

            if isinstance(recommender_object, ItemKNNCFRecommender):
                fit_params = {"topK": itemkNN_best_HP["topK"], "similarity": itemkNN_best_HP["similarity"], "shrink": itemkNN_best_HP["shrink"], "normalize": itemkNN_best_HP["normalize"]}
            
            elif isinstance(recommender_object, UserKNNCFRecommender):
                fit_params = {"topK": userkNN_best_HP["topK"],  "similarity": userkNN_best_HP["similarity"] , "shrink": itemkNN_best_HP["shrink"], "normalize": itemkNN_best_HP["normalize"]}

            elif isinstance(recommender_object, P3alphaRecommender):
                fit_params = {"topK": RP3alpha_best_HP["topK"], "alpha": RP3alpha_best_HP["alpha"], "normalize_similarity": RP3alpha_best_HP["normalize_similarity"]}
            
            elif isinstance(recommender_object, RP3betaRecommender):
                fit_params = {"topK": RP3beta_best_HP["topK"], "alpha": RP3beta_best_HP["alpha"], "beta": RP3beta_best_HP["beta"], "normalize_similarity": RP3beta_best_HP["normalize_similarity"]}
            else: # get defaut parameters...........
                fit_params = {}
            # measure training time.....
            start = time.time()
            recommender_object.fit(**fit_params)
            training_time = time.time() - start
            # testing for all records.....
            start = time.time()
            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)
            testing_time = time.time() - start
            averageTestingForOneRecord = testing_time / len(URM_test.getnnz(axis=1) > 0) # get number of non-zero rows in test data
            
            results_run_1["TrainingTime(s)"] = [training_time] + [0 for i in range(results_run_1.shape[0] - 1)]
            results_run_1["TestingTimeforRecords(s)"] = [testing_time] + [0 for i in range(results_run_1.shape[0] - 1)]
            results_run_1["AverageTestingTimeForOneRecord(s)"] = [averageTestingForOneRecord] + [0 for i in range(results_run_1.shape[0] - 1)]

            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            results_run_1["cuttOff"] = results_run_1.index
            results_run_1.insert(0, 'cuttOff', results_run_1.pop('cuttOff'))


            if resolveLastFMDataLeakageIssue == True and args.dataset == "lastFm":
                results_run_1.to_csv(saved_results+"/"+args.dataset+"_resolveDataLeakageIssue_"+recommender_class.RECOMMENDER_NAME+".txt", sep = "\t", index = False)
            else:
                results_run_1.to_csv(saved_results+"/"+args.dataset+"_"+recommender_class.RECOMMENDER_NAME+".txt", sep = "\t", index = False)
        
        except Exception as e:
            traceback.print_exc()

    


    


    


    


