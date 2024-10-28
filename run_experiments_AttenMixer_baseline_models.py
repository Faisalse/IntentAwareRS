from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from AttenMixer.digi_preprocessing import *
from AttenMixer.lastfm_preprocessing import *
from AttenMixer.gowalla_preprocessing import *
from AttenMixer.rsc15 import *
from AttenMixer.retail_rocket import *
from AttenMixer.baselines.sfcknn.main_sfcknn import *
from AttenMixer.baselines.vstan.main_vstan import *
from AttenMixer.baselines.stan.main_stan import *
from AttenMixer.models import SessionGraphAttn
from AttenMixer.baselines.SR.main_sr import *
from AttenMixer.dataset import SessionData
from AttenMixer.accuracy_measures import *
import pytorch_lightning as pl
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='retailrocket',help='dataset name: diginetica/gowalla/lastfm/yoochoose1_4/retailrocket/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', type=bool, default=False, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,help='split the portion of training set as validation set')
parser.add_argument('--alpha', type=float, default=0.75, help='parameter for beta distribution')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--use_lp_pool', type=str, default="True")
parser.add_argument('--train_flag', type=str, default="True")
parser.add_argument('--PATH', default='../checkpoint/Atten-Mixer_gowalla.pt', help='checkpoint path')
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--softmax', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--last_k', type=int, default=7)
parser.add_argument('--l_p', type=int, default=7)
parser.add_argument('--topk', type=float, default=[10, 20], help='learning rate')
parser.add_argument_group()
opt = parser.parse_args()
data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()
 
MRR_dictionary = dict()
for i in opt.topk:
    MRR_dictionary["MRR_"+str(i)] = MRR(i)      
HR_dictionary = dict()
for i in opt.topk:
    HR_dictionary["HR_"+str(i)] = HR(i)



hyperparameter_defaults = vars(opt)
config = hyperparameter_defaults
class AreaAttnModel(pl.LightningModule):
    def __init__(self, opt, n_node):
        super().__init__()
        self.opt = opt
        self.cnt = 0
        self.best_res = [0, 0]
        self.model = SessionGraphAttn(opt, n_node)
        self.loss = nn.Parameter(torch.Tensor(1))

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()
        hidden = self(items)
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        loss = self.model.loss_function(scores, targets - 1)
        return loss

    def test_step(self, batch, idx):
        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()
        hidden = self(items)
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = sub_scores.cpu().detach().numpy()
        
        for score, target in zip(sub_scores, targets):
            tar = target - 1
            recommendation_list = pd.Series([0 for i in range(len(score))], index = score)
            
            for key in MRR_dictionary:
                MRR_dictionary[key].add(recommendation_list, tar)
            
            for key in HR_dictionary:
                HR_dictionary[key].add(recommendation_list, tar)

    def on_test_epoch_end(self):
        print("on_test_epoch")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_dc_step, gamma=opt.lr_dc)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def main():
    seed = 123
    pl.seed_everything(seed)
    if opt.dataset == 'diginetica':
        
        name = "train-item-views.csv"
        obj1 = DIGI()
        train_features, test_features, n_node, original_train, original_test, train_validation, test_validation,  word2index, index2word =  obj1.data_load(data_path / name)
        opt.heads      = 4

    elif opt.dataset == 'gowalla':
        print("gowalla dataset")
        name = "loc-gowalla_totalCheckins.txt.gz"
        obj1 = Gowalla()
        train_features, test_features, n_node, original_train, original_test, train_validation, test_validation,  word2index, index2word =  obj1.data_load(data_path / name)
        opt.heads      = 6
    
    elif opt.dataset == 'lastfm':
        print("lastfm dataset")
        name = "userid-timestamp-artid-artname-traid-traname.tsv"
        obj1 = LastFm()
        train_features, test_features, n_node, _, _, _, _ =  obj1.data_load(data_path / name)
        print("Number of unique artist:  ", n_node)
        opt.heads      = 1

    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':

        name = "yoochoose-clicks.dat"
        obj1 = Data_processing(dataset = opt.dataset, path = data_path / name)
        tra_sess, tes_sess, sess_clicks = obj1.data_load()
        tra_ids, tra_dates, tra_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
        tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
        tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs_train(tra_seqs, tra_dates)
        te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs_test(tes_seqs, tes_dates)
        n_node = len(obj1.item_dict) + 1
        train_features = [tr_seqs, tr_labs]
        test_features = [te_seqs, te_labs]
        opt.batchSize  = 256
        opt.hiddenSize = 100
        opt.epoch      = 30
        opt.lr         =    0.0078
        opt.l2         =    0.0000628
        opt.heads      = 4

    elif opt.dataset == "retailrocket":
        name = "events.csv"
        data =  load_data_retail(data_path / name)
        data = filter_data(data)
        train, test = split_data_only(data)
        train_val, test_val = validation_data(train)
        train_features, test_features, n_node  = data_augmentation_transformation_for_Dl(train_val, test_val)

        opt.batchSize  = 256
        opt.hiddenSize = 100
        opt.epoch      = 1
        opt.lr         =    0.0081
        opt.l2         =    0.000199
        opt.heads      = 4

    session_data = SessionData(train_features, test_features, batch_size=opt.batchSize)
    trainer = pl.Trainer(deterministic=True, max_epochs= opt.epoch, num_sanity_val_steps=2)
    model = AreaAttnModel(opt=opt, n_node=n_node)
    print("Model fitting")
    trainer.fit(model, session_data)
    print("Model evaluation starts")
    model.eval()
    trainer.test(model, session_data.test_dataloader(test_features))
    result_frame = pd.DataFrame()    
    for key in MRR_dictionary:
        print(key +"   "+ str(  MRR_dictionary[key].score()    ))
        result_frame[key] =   [MRR_dictionary[key].score()]
    # get the results of HR values.....    
    for key in HR_dictionary:
        print(key +"   "+ str(  HR_dictionary[key].score()    ))
        result_frame[key] = [HR_dictionary[key].score()]

    name = "Atten_Mixer_"+opt.dataset+".txt"
    print(result_frame)
    result_frame.to_csv(result_path/name, sep='\t', index = False) 


if __name__ == "__main__":
    #obj = SequentialRulesMain(data_path, result_path, dataset = opt.dataset)
    #obj.fit_(opt.topk, opt.topk)
    
    #obj = SFCKNN_MAIN(data_path, result_path, dataset = opt.dataset)
    #obj.fit_(opt.topk, opt.topk)
    
    #obj = STAN_MAIN(data_path, result_path, dataset = opt.dataset)
    #obj.fit_(opt.topk, opt.topk)s
    obj = VSTAN_MAIN(data_path, result_path, dataset = opt.dataset)
    obj.fit_(opt.topk, opt.topk)
    #main()
    
    