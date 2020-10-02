import numpy as np
import pandas as pd
import os
from os import walk
import pickle
import collections
from sklearn.model_selection import train_test_split
from ATTfold.common.utils import get_pairings

dataset = 'rnastralign'
rna_types = ['5S_rRNA','16S_rRNA','group_I_intron','RNaseP','SRP',
              'telomerase','tmRNA','tRNA']
length_limit = 512
seed = 8

data_path = '../raw_data/RNAStrAlign'
data_file_list = list()


for rna_type in rna_types:
    type_dir = os.path.join(data_path, rna_type+'_database')
    for r,d,f in walk(type_dir):
        for file in f:
            if file.endswith('.ct'):
                data_file_list.append(os.path.join(r,file))

data_list = list(map(lambda x:pd.read_csv(x,
                sep='\s+', skiprows=1,header=None),data_file_list))

seq_len_list = list(map(len,data_list))

file_len_dic = dict()
for i in range(len(seq_len_list)):
    file_len_dic[data_file_list[i]] = seq_len_list[i]


data_list = list(filter(lambda x:len(x)<=length_limit,data_list))
seq_len_list = list(map(len, data_list))
data_file_list = list(filter(lambda x: file_len_dic[x]<=length_limit, data_file_list))

pairs_list = list(map(get_pairings, data_list))

def generate_label(data):
    rnadata1 = data.loc[:,0]
    rnadata2 = data.loc[:,4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)

structure_list = list(map(generate_label,data_list))
seq_list = list(map(lambda x:''.join(list(x.loc[:,1])),data_list))

label_dict = {
    '.': np.array([1,0,0]),
    '(': np.array([0,1,0]),
    ')': np.array([0,0,1])
}
seq_dict = {
    'A':np.array([1,0,0,0]),
    'U':np.array([0,1,0,0]),
    'C':np.array([0,0,1,0]),
    'G':np.array([0,0,0,1]),
    'N':np.array([0,0,0,0])
}

def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x:seq_dict[x], str_list))
    return np.stack(encoding,axis=0)

def stru_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x:label_dict[x],str_list))
    return np.stack(encoding,axis=0)

def padding(data_array,maxlen):
    a,b = data_array.shape
    return np.pad(data_array,((0,maxlen-a),(0,0)), 'constant')

max_len = max(seq_len_list)
seq_encoding_list = list(map(seq_encoding,seq_list))
stru_encoding_list = list(map(stru_encoding,structure_list))

seq_encoding_list_padded = list(map(lambda x: padding(x, max_len),
                                seq_encoding_list))
stru_encoding_list_padded = list(map(lambda x:padding(x, max_len),
                                     stru_encoding_list))

RNA_SS_data = collections.namedtuple('RNA_SS_data',
                                       'seq stru length name pairs')
RNA_SS_data_list = list()

for i in range(len(data_list)):
    RNA_SS_data_list.append(RNA_SS_data(seq=seq_encoding_list_padded[i],
                                            stru=stru_encoding_list_padded[i],
                                            length=seq_len_list[i],
                                            name=data_file_list[i],
                                            pairs=pairs_list[i]))
# training test split （0.8 train   0.1 val   0.1 test)
RNA_SS_train,RNA_SS_test = train_test_split(RNA_SS_data_list,
                                            test_size=0.2,random_state=seed)
RNA_SS_test,RNA_SS_val = train_test_split(RNA_SS_test,
                                          test_size=0.5,random_state=seed)

savepath = dataset+'_512'
os.mkdir(savepath)

for i in ['train','test','val']:
    with open(savepath+'/%s.pickle' % i,'wb') as f:
        pickle.dump(eval('RNA_SS_'+i),f)
