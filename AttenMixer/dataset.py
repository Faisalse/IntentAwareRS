import os
import pickle
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler, SequentialSampler
from AttenMixer.utils import split_validation, data_masks, Data
import pytorch_lightning as pl

class SessionData(pl.LightningDataModule):
    
    def __init__(self, train_data, test_data , validation=False, batch_size=100):
        super(SessionData, self).__init__()
        self.validation = validation
        self.batch_size = batch_size
        
        self.train_data = train_data
        self.test_data = test_data

    def setup(self, stage=None):
        
        if stage == 'fit' or stage is None:
           
            if self.validation:
                self.train_data, self.valid_data = split_validation(self.train_data, 0.1)
            else:
                self.valid_data = self.test_data
            self.train_data = Data(self.train_data, shuffle=True)
            self.valid_data = Data(self.valid_data, shuffle=True)


    def train_dataloader(self):
        sampler = BatchSampler(SequentialSampler(self.train_data), batch_size=self.batch_size, drop_last=False)
        return DataLoader(self.train_data, sampler=sampler, num_workers=0, pin_memory=True)
    
    def val_dataloader(self):
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.valid_data))), batch_size=self.batch_size, drop_last=False)
        return DataLoader(self.valid_data, sampler=sampler, num_workers=0, pin_memory=True)

    def test_dataloader(self, test_data):
        self.test_data = Data(test_data, shuffle=True)
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.test_data))), batch_size=self.batch_size, drop_last=False)
        return DataLoader(self.test_data, sampler=sampler, num_workers=0, pin_memory=True)
        

if __name__ == "__main__":

    data = SessionData()
    data.setup()
    val_loader = data.val_dataloader()
    train_loader = data.train_dataloader()
    
    
    
    
