from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from AttenMixer.digi_preprocessing import *
from AttenMixer.lastfm_preprocessing import *
from AttenMixer.gowalla_preprocessing import *
from AttenMixer.rsc15 import *
from AttenMixer.retail_rocket import *
from AttenMixer.models import SessionGraphAttn
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
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64',help='dataset name: diginetica/gowalla/lastfm/ yoochoose1_64/ retailrocket')
parser.add_argument('--numberOfIteration', type=int, default=30, help='Number of iterations for training....')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=1, help='the number of epochs to train for')
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
parser.add_argument('--topK', type=float, default=[10, 20]) 
parser.add_argument_group()
opt = parser.parse_args()
data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()
 
Performance_measures = dict()
for i in opt.topK:
    Performance_measures["MRR_"+str(i)] = MRR(i)
    Performance_measures["HR_"+str(i)] = HR(i)      

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.epoch = 0
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epoch = epoch
            self.counter = 0

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

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
            for key in Performance_measures:
                Performance_measures[key].add(recommendation_list, tar)
            

    def on_test_epoch_end(self):
        print("on_test_epoch")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_dc_step, gamma=opt.lr_dc)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def get_validation_data(trining_data, ratio = 0.2):
     train_x, train_y = trining_data[0], trining_data[1]
     test_records = int(len(trining_data[0]) * ratio)
     train_tr = [train_x[ : -test_records], train_y[ : -test_records]   ]
     train_val = [train_x[ -test_records : ], train_y[ -test_records : ]   ]
     return train_tr, train_val

def main():
    seed = 123
    pl.seed_everything(seed)
    if opt.dataset == 'diginetica':
        
        name = "train-item-views.csv"
        obj1 = DIGI()
        train_features, test_features, n_node, original_train, original_test, train_validation, test_validation,  word2index, index2word =  obj1.data_load(data_path / name)

    elif opt.dataset == 'gowalla':
        name = "loc-gowalla_totalCheckins.txt.gz"
        obj1 = Gowalla()
        train_features, test_features, n_node, original_train, original_test, train_validation, test_validation,  word2index, index2word =  obj1.data_load(data_path / name)
        
    elif opt.dataset == 'lastfm':
        name = "userid-timestamp-artid-artname-traid-traname.tsv"
        obj1 = LastFm()
        train_features, test_features, n_node, _, _, _, _ =  obj1.data_load(data_path / name)

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
        train_features, test_features = get_validation_data(train_features, ratio=0.10)

    elif opt.dataset == "retailrocket":
        name = "events.csv"
        data =  load_data_retail(data_path / name)
        data = filter_data(data)
        train, test = split_data_only(data)
        train_val, test_val = validation_data(train)
        train_features, test_features, n_node  = data_augmentation_transformation_for_Dl(train_val, test_val)

    # model optimization......
    for i in range(opt.numberOfIteration):
        for key in Performance_measures:
                Performance_measures[key].reset()
        
        #opt.hiddenSize   = random.choice([50, 100, 150, 200])
        opt.epoch   = random.choice([10, 20, 30])
        opt.lr      =    random.uniform(5e-4, 2e-2)
        opt.l2      =    random.uniform(5e-5, 2e-3)
        opt.heads   = random.choice([1, 2, 4, 8, 16])

        session_data = SessionData(train_features, test_features, validation = opt.validation, batch_size=opt.batchSize)
        trainer = pl.Trainer(deterministic=True, max_epochs= opt.epoch, num_sanity_val_steps=2,)
        model = AreaAttnModel(opt=opt, n_node=n_node)
        trainer.fit(model, session_data)
        trainer.test(model, session_data.test_dataloader(test_features))
        name = "Atten_Mixer_optimize_"+opt.dataset+".txt"
        result_frame = pd.DataFrame()    
        for key in Performance_measures:
            print(key +"   "+ str(  Performance_measures[key].score()    ))
            result_frame[key] =   [Performance_measures[key].score()]
            print(len(Performance_measures[key].score_))
        result_frame["epoch"] = [opt.epoch]
        result_frame["hiddenSize"] = opt.hiddenSize
        result_frame["lr"] = [opt.lr]
        result_frame["l2"] =  [opt.l2]
        result_frame["heads"] =  [opt.heads]
        
        if os.path.exists(result_path/name):
            old = pd.read_csv(result_path/name, sep='\t')
            updadated = pd.concat([old, result_frame])
            updadated.to_csv(result_path/name, sep='\t', index = False)
        else:
            print(result_frame)
            result_frame.to_csv(result_path/name, sep='\t', index = False)
if __name__ == "__main__":
    main()
    
    