"""
Created on July, 2018

@author: Tangrizzly
"""
from MCPRN.utils import Data, split_validation
from MCPRN.rsc15_data_preprocessing import *
from MCPRN.tmall_data_preprocessing import *
from MCPRN.accuracy_measures import *
from  pathlib import Path
from MCPRN.model import *
import argparse
import torch

# baseline models...
from MCPRN.baselines.sfcknn.main_sfcknn import *
from MCPRN.baselines.vstan.main_vstan import *
from MCPRN.baselines.stan.main_stan import *
from MCPRN.baselines.SR.main_sr import *

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: diginetica/yoochoose/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--train_batchSize', type=int, default=50, help='train input batch size')
parser.add_argument('--test_batchSize', type=int, default=50, help='test input batch size')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--n_interest', type=int, default=3, help='the number of interests')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--topKList', type=float, default=[5, 10, 20], help='top_KList')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--debug', default=False)
parser.add_argument('--loss', default='PCE')
opt = parser.parse_args()

data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()
accuracy_dictionary = dict()
for i in opt.topKList:
    accuracy_dictionary["MRR_"+str(i)] = MRR(i)
    accuracy_dictionary["NDCG_"+str(i)] = NDCG(i)

def main():
    print(opt)
    if opt.dataset == 'yoochoose':
        name = "yoochoose-buys.dat"
        dataset = load_data_rsc15(data_path / name)
        filter_data_ = filter_data_rsc15(dataset)
        train_data_full, test_data, item_no, _, _ = split_data_rsc15(filter_data_)
        val_train_data, valid_data = split_validation(train_data_full, opt.valid_portion)
        n_node = item_no
    elif opt.dataset == 'tmall':
        name = "dataset15.csv"
        dataset = load_data_tmall(data_path / name)
        filter_data_ = filter_data_tmall(dataset)
        train_data_full, test_data, item_no = split_data_tmall(filter_data_)
        val_train_data, valid_data = split_validation(train_data_full, opt.valid_portion)
        n_node = item_no
    val_train_data = Data(val_train_data, shuffle=True, n_node = n_node)
    valid_data = Data(valid_data, shuffle=False ,n_node = n_node)

    train_data_full = Data(train_data_full, shuffle=True, n_node = n_node)
    test_data = Data(test_data, shuffle=False ,n_node = n_node)
    model = trans_to_cuda(SessionGraph(opt, n_node, opt.n_interest))

    #print("Model training on validation data to get best epoch values")
    #best_epoch = train_validation(opt.epoch, model, val_train_data, valid_data, accuracy_dictionary, opt.patience)
    #print("Best Epoch Value:   "+str(best_epoch))
    print("Modeling training on full data")
    train_test(30, model, train_data_full, test_data, accuracy_dictionary)

    for key in accuracy_dictionary:
        print(key +":  "+  str( accuracy_dictionary[key].score()  ))

        result_frame = pd.DataFrame()    
    for key in accuracy_dictionary:
        print(key +"   "+ str(  accuracy_dictionary[key].score()))
        result_frame[key] =   [accuracy_dictionary[key].score()]

    name = "MCRPN_"+opt.dataset+".txt"
    print(result_frame)
    result_frame.to_csv(result_path/name, sep='\t', index = False) 



if __name__ == '__main__':
    #main()
    print("Experiments are runinig for SR model................... wait for results...............")
    se_obj = SequentialRulesMain(data_path, result_path, dataset = opt.dataset)
    se_obj.fit_(opt.topKList)

    print("Experiments are runinig for STAN model................... wait for results...............")
    stan_obj = STAN_MAIN(data_path, result_path, dataset = opt.dataset)
    stan_obj.fit_(opt.topKList)

    print("Experiments are runinig for VSTAN model................... wait for results...............")
    vstan_obj = VSTAN_MAIN(data_path, result_path, dataset = opt.dataset)
    vstan_obj.fit_(opt.topKList)
    
    print("Experiments are runinig for SFCKNN model................... wait for results...............")
    sfcknn_obj = SFCKNN_MAIN(data_path, result_path, dataset = opt.dataset)
    sfcknn_obj.fit_(opt.topKList)
    
