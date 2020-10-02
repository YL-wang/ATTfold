import os
import pickle
import collections
from ATTfold.common.utils import *
from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
import torch

class RNADataProcessor(object):
    def __init__(self, data_dir, split, upsampling=False):
        self.data_dir = data_dir  
        self.split = split          
        self.upsampling = upsampling 
        self.load_data()

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data',
            'seq ss_label length name pairs')
        with open(os.path.join(data_dir, '%s.pickle' % self.split), 'rb') as f:
            self.data = pickle.load(f)
        if self.upsampling:
            self.data = self.upsampling_data()
        self.data_x = np.array([instance[0] for instance in self.data])  # seq =([[1,0,0,0],[0,0,1,0]]) #(序列个数，序列maxlen,4)
        self.data_y = np.array([instance[1] for instance in self.data])  # stru = ([[0,1,0],[0,1,0]])#(序列个数，序列maxlen,3)
        self.pairs = np.array([instance[-1] for instance in self.data])  # pairs = [[0,69],[1,68]]
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data) 
        self.seq = list(p.map(encoding2seq, self.data_x))  
        self.seq_max_len = len(self.data_x[0]) 

    def upsampling_data(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split('/')[3], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()

        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]
            data_list.append(data[index])
        final_d_list = list()

        for i in [0,2,3,4,6]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], int(max_num/2))
            final_d_list += list(d[index])

        for i in [1,5,7]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], int(max_num * 2))
            final_d_list += list(d[index])


        shuffle(final_d_list)
        return final_d_list

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact


    def get_one_sample(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]

        contact= self.pairs2map(data_pair)
        return contact, data_seq, data_len


class Dataset(data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.len

    def __getitem__(self, index):
        # Select sample
        return self.data.get_one_sample(index)




