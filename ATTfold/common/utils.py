import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import random
import os

label_dict = {
    '.': np.array([1, 0, 0]),
    '(': np.array([0, 1, 0]),
    ')': np.array([0, 0, 1])
}
seq_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0])
}

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}

def get_args():
    argparser = argparse.ArgumentParser(description="diff through ATTfold")
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='./config.json',
        help='The Configuration file'
    )
    argparser.add_argument('--test', type=bool, default=False,
                           help='skip training to test directly.')
    args = argparser.parse_args()
    return args


def soft_sign(x):
    return torch.sigmoid(x)


def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('.')
        else:
            seq.append(char_dict[np.argmax(arr_row)])
    return ''.join(seq)


def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')

def F1_low_tri(opt_state, true_a):
    tril_index = np.tril_indices(len(opt_state), k=-1)
    return f1_score(true_a[tril_index], opt_state[tril_index])


def acc_low_tri(opt_state, true_a):
    tril_index = np.tril_indices(len(opt_state), k=-1)
    return accuracy_score(true_a[tril_index], opt_state[tril_index])


def evaluate_exact(pred_a, true_a):
    tp_map = torch.sign(torch.Tensor(pred_a) * torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision*recall) / (precision+recall)
    return precision, recall, f1_score


def contact_map_masks(seq_lens, max_len): 
    n_seq = len(seq_lens)
    masks = np.zeros([n_seq, max_len, max_len])
    for i in range(n_seq):
        l = int(seq_lens[i].cpu().numpy())
        masks[i, :l, :l] = 1
    return masks


# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:, 0].values)
    rnadata2 = list(data.loc[:, 4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1] > 0, rna_pairs))
    rna_pairs = (np.array(rna_pairs) - 1).tolist()
    return rna_pairs

#get position embedding
def get_pe(seq_lens, max_len): 

    num_seq = seq_lens.shape[0] 

    pos_i_abs = torch.Tensor(np.arange(1, max_len + 1)).view(1, -1, 1).expand(num_seq, -1, -1).double()
    pos_i_rel = torch.Tensor(np.arange(1, max_len + 1)).view(1, -1).expand(num_seq, -1) 
    pos_i_rel = pos_i_rel.double() / seq_lens.view(-1, 1).double()
    pos_i_rel = pos_i_rel.unsqueeze(-1)
    
    pos = torch.cat([pos_i_abs, pos_i_rel], -1)

    PE_element_list = list()

    PE_element_list.append(pos) 
    PE_element_list.append(1.0 / pos_i_abs) 
    PE_element_list.append(1.0 / torch.pow(pos_i_abs, 2))

    for n in range(1, 50):
        PE_element_list.append(torch.sin(n * pos))

    for i in range(2, 5):
        PE_element_list.append(torch.pow(pos_i_rel, i))

    for i in range(3):
        gaussian_base = torch.exp(-torch.pow(pos,
                                             2)) * math.sqrt(math.pow(2, i) / math.factorial(i)) * torch.pow(pos, i)
        PE_element_list.append(gaussian_base)

    PE = torch.cat(PE_element_list, -1)
    for i in range(num_seq):
        PE[i, seq_lens[i]:, :] = 0
    return PE

def f1_loss(pred_a, true_a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_a  = -(F.relu(-pred_a+1)-1)

    true_a = true_a.unsqueeze(1)
    unfold = nn.Unfold(kernel_size=(3, 3), padding=1)
    true_a_tmp = unfold(true_a)
    w = torch.Tensor([0, 0.0, 0, 0.0, 1, 0.0, 0, 0.0, 0]).to(device)
    true_a_tmp = true_a_tmp.transpose(1, 2).matmul(w.view(w.size(0), -1)).transpose(1, 2)
    true_a = true_a_tmp.view(true_a.shape)
    true_a = true_a.squeeze(1)

    tp = pred_a*true_a
    tp = torch.sum(tp, (1,2))

    fp = pred_a*(1-true_a)
    fp = torch.sum(fp, (1,2))

    fn = (1-pred_a)*true_a
    fn = torch.sum(fn, (1,2))

    f1 = torch.div(2*tp, (2*tp + fp + fn))
    return 1-f1.mean()

def seed_torch(seed=8):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True






