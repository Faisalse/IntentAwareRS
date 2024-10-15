import sys
import time
import argparse
import pickle
import os
import logging
from HIDE.hide.sessionG import *
from HIDE.hide.utils import *
from HIDE.datasets.process_tmall_class import *
from pathlib import Path
from HIDE.accuracy_measures import *
from HIDE.baselines.SR.main_sr import *
from HIDE.baselines.vstan.main_vstan import *
from HIDE.baselines.stan.main_stan import *
from HIDE.baselines.sfcknn.main_sfcknn import *

import pandas as pd
import numpy as np
import time
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='diginetica/Nowplaying/tmall')
parser.add_argument('--model', default='HIDE', help='[GCEGNN, SRGNN, DHCN, SAHNN, COTREC]')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--w', type=int, default=6, help='max window size')
parser.add_argument('--n_factor', type=int, default=5, help='Disentangle factors number')
parser.add_argument('--gpu_id', type=str,default="0")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used')
parser.add_argument('--n_iter', type=int, default=1)    
parser.add_argument('--seed', type=int, default=2021)                                 
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.') 
parser.add_argument('--dropout_global', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--e', type=float, default=0.4, help='Disen H sparsity.')
parser.add_argument('--disen', action='store_true', help='use disentangle')
parser.add_argument('--lamda', type=float, default=1e-4, help='aux loss weight')
parser.add_argument('--norm', default= True, action='store_true', help='use norm')
parser.add_argument('--sw_edge', default= True,  action='store_true', help='slide_window_edge')
parser.add_argument('--item_edge', default= True,  action='store_true', help='item_edge')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--topKList', type=float, default=[1, 5, 10, 20, 50, 100], help='learning rate')
parser.add_argument('--original', type=bool, default=False, help='learning rate')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()

def run_experiments_for_HIDE():
    print("<<<<<<<<<<<<<<<<  "+opt.dataset+" >>>>>>>>>>>>>>>>>>>>")
    exp_seed = opt.seed
    init_seed(exp_seed)
    sw = []
    for i in range(2, opt.w+1):
        sw.append(i)
    if opt.original == True:
        datapath = Path("HIDE/HIDE_without_AnyChanges/datasets/"+opt.dataset)
        datapath = datapath.resolve()
        all_train_seq = "all_train_seq.txt"
        train = "train.txt"
        test = "test.txt"
        all_train = pickle.load(open(datapath / all_train_seq, 'rb'))
        train_data = pickle.load(open(datapath / train, 'rb'))
        test_data = pickle.load(open(datapath / test, 'rb'))   
        num_node = 40727
    else:
        if opt.dataset == 'tmall':
            name = "dataset15.csv"
            obj1 = Tmall()
            tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)
            tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
            tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
            num_node = len(obj1.item_dict) + 1
            tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
            # test sequences...
            te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
            num_node = num_node
            opt.n_iter = 1
            opt.dropout_gcn = 0.6
            opt.dropout_local = 0.0
            opt.e = 0.4
            opt.w = 6
            opt.nonhybrid = True
            sw = []
            for i in range(2, opt.w+1):
                sw.append(i)
            train_data = [tr_seqs, tr_labs]
            test_data = [te_seqs, te_labs]

        elif opt.dataset == 'lastfm':
            num_node = 35231
            opt.n_iter = 1
            opt.dropout_gcn = 0.1
            opt.dropout_local = 0.0
        else:
            num_node = 310
    print("SEED VALUE:{}".format(exp_seed))
    # ==============================
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = test_data
    train_data = Data(train_data, opt, n_node=num_node, sw=sw)
    test_data = Data(test_data, opt, n_node=num_node, sw=sw)

    if opt.model == 'HIDE':
        model = trans_to_cuda(HIDE(opt, num_node))
    start = time.time()
    print("*************** Training *******************")
    start = time.time()    
    model = train_test(model, train_data, opt)  
    end = time.time()
    totalTime = (end - start) / 3600
    print(f'TIME REQUIRED TO TRAIN HIDE MODEL:  {totalTime}')
    print("*************** Predicting *******************")
    # intialize MRR class
    performance_measures = dict()
    for i in opt.topKList:
        performance_measures["MRR_"+str(i)] = MRR(i)
        performance_measures["Pre_"+str(i)] = Precision(i)
        performance_measures["HR_"+str(i)] = HR(i)
    
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=1,
                                              shuffle=False, pin_memory=True)
    model.eval()
    for data in test_loader:
        target, scores = forward(model, data)
        target = target.numpy()
        index_ = scores.topk(20)[1]
        index_ = trans_to_cpu(index_).detach().numpy()
        index_ = index_.flatten()
        
        score_ = scores.topk(20)[0]
        score_ = trans_to_cpu(score_).detach().numpy()
        score_ = score_.flatten()
        print("Maximum score:       "+ str(max(score_)))
        target = target[0] - 1
        recommendation_list = pd.Series(score_, index = index_)
        recommendation_list = recommendation_list.sort_values(ascending=False)
        # Calculate the MRR values
        for key in performance_measures:
            performance_measures[key].add(recommendation_list, target)     
    # get the results of MRR values.....
    result_frame = pd.DataFrame()    
    for key in performance_measures:
        print(key +"   "+ str(  performance_measures[key].score()    ))
        result_frame[key] =   [performance_measures[key].score()]
          
    name = "HIDE_"+opt.dataset+".txt"
    result_frame.to_csv(result_path/name, sep = "\t", index = False) 
 
if __name__ == '__main__':
    

    
    print("SR model")
    se_obj = SequentialRulesMain(data_path, result_path, dataset = opt.dataset)
    se_obj.fit_(opt.topKList)
    
    # # VSTAN method........
    print("VSTAN model")
    vstan_obj = VSTAN_MAIN(data_path, result_path, dataset = opt.dataset)
    vstan_obj.fit_(opt.topKList)
    
    # # stan.....
    print("STAN model")
    stan_obj = STAN_MAIN(data_path, result_path, dataset = opt.dataset)
    stan_obj.fit_(opt.topKList)
    
    # # SFCKNN
    print("SFCKNN model")
    sfcknn_obj = SFCKNN_MAIN(data_path, result_path, dataset = opt.dataset)
    sfcknn_obj.fit_(opt.topKList)

    run_experiments_for_HIDE()
    
    
    






















