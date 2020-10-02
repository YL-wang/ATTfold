import numpy as np
import pandas as pd
import os
from os import walk
import collections
from collections import defaultdict
import seaborn as sns
import pickle


dataset = 'rnastralign'
rna_types = ['5S_rRNA','16S_rRNA','group_I_intron','RNaseP','SRP',
             'telomerase','tmRNA','tRNA']


data_path = '../raw_data/RNAStrAlign'
seed = 8

file_list = list()


for rna_type in rna_types:
    type_dir = os.path.join(data_path, rna_type+'_database')
    for r,d,f in walk(type_dir):
        for file in f:
            if file.endswith('.ct'):
                file_list.append(os.path.join(r,file))

data_list = list(map(lambda x:pd.read_csv(x,
                sep='\s+', skiprows=1,header=None),file_list))

seq_list = list(map(lambda x: ''.join(list(x.loc[:,1])),data_list))

seq_file_pair_list = list(zip(seq_list, file_list)) #(序列，文件名）

d = defaultdict(list)
for k,v in seq_file_pair_list:
    d[k].append(v)

unique_seqs = list()
seq_files = list()

for k,v in d.items():
    unique_seqs.append(k) 
    seq_files.append(v)  

original_seq_len = list(map(len, seq_list)) 
unique_seq_len = list(map(len,unique_seqs)) 
cluseter_size = list(map(len,seq_files)) 
used_files = list(map(lambda x:x[0],seq_files)) 

used_files_rna_type = list(map(lambda x: x.split('/')[3],used_files))


RNA_SS_data = collections.namedtuple('RNA_SS_data',
                                       'seq stru length name pairs')
with open('rnastralign_512/test.pickle','rb') as f:
    test_all = pickle.load(f)
with open('rnastralign_512/train.pickle','rb') as f:
    train_all = pickle.load(f)

file_seq_d = dict() 
for k,v in seq_file_pair_list:
    file_seq_d[v] = k

train_files = [instance.name for instance in train_all] 
train_seqs = [file_seq_d[file] for file in train_files] 
train_in_files = list()
for seq in train_seqs:
    files_tmp = d[seq] 
    train_in_files += files_tmp
train_in_files = list(set(train_in_files))

test_files = [instance.name for instance in test_all]
test_set = list(set(test_files)-set(test_files).intersection(train_in_files)) 
test_seqs = [file_seq_d[file] for file in test_set] 
test_seq_file_pair_list = zip(test_seqs,test_set)
test_seq_file_d =defaultdict(list)

for k,v in test_seq_file_pair_list:
    test_seq_file_d[k].append(v)
test_files_used = [test_seq_file_d[seq][0] for seq in test_seqs]

test_rna_type = list(map(lambda x:x.split('/')[3],test_files_used))


test_all_used = list()
for instance in test_all:
    if instance.name in test_files_used:
        test_all_used.append(instance)

with open('rnastralign_512/test_no_redundant_512.pickle', 'wb') as f:
    pickle.dump(test_all_used,f)

