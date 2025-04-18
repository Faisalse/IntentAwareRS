import pickle
import numpy as np
from time import time
from tqdm import tqdm
import scipy.sparse as sp
import random
from scipy.sparse import coo_matrix

class Data(object):
    def __init__(self, args, validation = False):

        self.path = args.data_path
        self.n_batch = args.n_batch
        self.batch_size = args.batch_size
        self.train_num = args.train_num
        self.sample_num = args.sample_num

        try:
            train_file = self.path / 'train.pkl'
            test_file = self.path / 'test.pkl'
            
            with open(train_file, 'rb') as f:
                train_mat = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_mat = pickle.load(f)
            
            if validation == True:
                print("data validation")
                train_mat, test_mat = self.split_sparse_matrix(train_mat)

        except Exception as e:
            print("Try an alternative way of reading the data.")
            train_file = self.path + '/train_index.pkl'
            test_file = self.path + '/test_index.pkl'

            with open(train_file, 'rb') as f:
                train_index = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_index = pickle.load(f)
            train_row, train_col = train_index[0], train_index[1]
            n_user = max(train_row) + 1
            n_item = max(train_col) + 1

            train_mat = sp.coo_matrix((np.ones(len(train_row)), (train_row, train_col)), shape=[n_user, n_item])
            test_row, test_col = test_index[0], test_index[1]
            test_mat = sp.coo_matrix((np.ones(len(test_row)), (test_row, test_col)), shape=[n_user, n_item])

        # get number of users and items
        self.n_users, self.n_items = train_mat.shape[0], train_mat.shape[1]
        self.n_train, self.n_test = len(train_mat.row), len(test_mat.row)

        self.print_statistics()

        self.R = train_mat.todok()
        self.train_items, self.test_set = {}, {}

        train_uid, train_iid = train_mat.row, train_mat.col
        for i in range(len(train_uid)):
            uid = train_uid[i]
            iid = train_iid[i]
            if uid not in self.train_items:
                self.train_items[uid] = [iid]
            else:
                self.train_items[uid].append(iid)
        test_uid, test_iid = test_mat.row, test_mat.col
        for i in range(len(test_uid)):
            uid = test_uid[i]
            iid = test_iid[i]
            if uid not in self.test_set:
                self.test_set[uid] = [iid]
            else:
                self.test_set[uid].append(iid)
    
    def split_sparse_matrix(self, matrix, test_ratio=0.20):
    # Ensure matrix is in COO format
        if not isinstance(matrix, coo_matrix):
            raise ValueError("Input matrix must be a scipy.sparse.coo_matrix")
        
        # Get row, col, and data from the matrix
        row, col, data = matrix.row, matrix.col, matrix.data
        
        # Initialize lists for train and test data
        train_row, train_col, train_data = [], [], []
        test_row, test_col, test_data = [], [], []
        
        # For each row, split data into train and test based on test_ratio
        unique_rows = np.unique(row)
        for r in unique_rows:
            # Get indices of the current row
            indices = np.where(row == r)[0]
            
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Determine split point
            test_size = int(len(indices) * test_ratio)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            
            # Append to test set
            test_row.extend(row[test_indices])
            test_col.extend(col[test_indices])
            test_data.extend(data[test_indices])
            
            # Append to train set
            train_row.extend(row[train_indices])
            train_col.extend(col[train_indices])
            train_data.extend(data[train_indices])
        
        # Create COO matrices for train and test
        train_matrix = coo_matrix((train_data, (train_row, train_col)), shape=matrix.shape)
        test_matrix = coo_matrix((test_data, (test_row, test_col)), shape=matrix.shape)
        
        return train_matrix, test_matrix

    def get_adj_mat(self):
        adj_mat = self.create_adj_mat()
        return adj_mat

    def create_adj_mat(self):
        t1 = time()
        rows = self.R.tocoo().row
        cols = self.R.tocoo().col
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr()

    def uniform_sample(self):
        users = np.random.randint(0, self.n_users, int(self.n_batch * self.batch_size))
        train_data = []
        for i, user in tqdm(enumerate(users), desc='Sampling Data', total=len(users)):
            pos_for_user = self.train_items[user]
            pos_index = np.random.randint(0, len(pos_for_user))
            pos_item = pos_for_user[pos_index]
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if self.R[user, neg_item] == 1:
                    continue
                else:
                    break
            train_data.append([user, pos_item, neg_item])
        self.train_data = np.array(train_data)
        return len(self.train_data)

    def mini_batch(self, batch_idx):
        st = batch_idx * self.batch_size
        ed = min((batch_idx + 1) * self.batch_size, len(self.train_data))
        batch_data = self.train_data[st: ed]
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items)))

    def get_statistics(self):
        sta = ""
        sta += 'n_users=%d, n_items=%d\t' % (self.n_users, self.n_items)
        sta += 'n_interactions=%d\t' % (self.n_train + self.n_test)
        sta += 'n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items))
        return sta
