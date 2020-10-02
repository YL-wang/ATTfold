import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch.optim as optim
from torch.utils import data

from ATTfold.data_generator import RNADataProcessor, Dataset
import collections
from ATTfold.ATTfold_model import ATTfold
from ATTfold.common.utils import *
from ATTfold.common.config import process_config
from ATTfold.evaluation import all_train,all_val,all_test

torch.multiprocessing.set_sharing_strategy('file_system')

args = get_args()

config_file = args.config
config = process_config(config_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = config.d
BATCH_SIZE = config.BATCH_SIZE
OUT_STEP = config.OUT_STEP
CU_steps = config.CU_steps
data_type = config.data_type
ATTfold_path = '../models_ckpt/ATTfold_d{}_s{}.pt'.format(d, CU_steps)
#ATTfold_path ='../models_ckpt/ATTfold_low_model.pt'
epoches = config.EPOCHES
save_epi = config.evaluate_save
steps_done = 0
val_tmp = 0
#LOAD_MODEL = True
LOAD_MODEL = False

seed_torch()

params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}

RNA_SS_data = collections.namedtuple('RNA_SS_data',
                                     'seq ss_label length name pairs')

train_data = RNADataProcessor('../data_preprocess/{}/'.format(data_type), 'train', True)
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_data = RNADataProcessor('../data_preprocess/{}/'.format(data_type), 'val')
val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)


params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 6,
          'drop_last': False}
test_data = RNADataProcessor('../data_preprocess/{}/'.format(data_type), 'test_no_redundant_512')
test_set = Dataset(test_data)
test_generator = data.DataLoader(test_set, **params)


seq_len = train_data.data_y.shape[-2]
print('Max seq length ', seq_len)

ATTfold = ATTfold(d=d, L=seq_len,
                   steps = CU_steps).to(device)

if LOAD_MODEL and os.path.isfile(ATTfold_path):
    print('Loading ATTfold model...')
    ATTfold.load_state_dict(torch.load(ATTfold_path))

# print(torch.cuda.device_count())
# if torch.cuda.device_count()>1:
#     print('3')
#     ATTfold = nn.DataParallel(ATTfold, device_ids=[0,1])
#    ATTfold.to(device)

all_optimizer = optim.Adam(ATTfold.parameters())


pos_weight = torch.Tensor([512]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight)

all_optimizer.zero_grad()
for epoch in range(epoches):
    ATTfold.train()
    result = list()
    result_shift = list()
    for contacts, seq_embeddings, seq_lens in train_generator:

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

        contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)

        PE_batch = get_pe(seq_lens, seq_len).float().to(device)

        pred_score, a_pred = ATTfold(PE_batch, seq_embedding_batch)

        loss_u = criterion_bce_weighted(pred_score * contact_masks, contacts_batch)

        loss_a = f1_loss(a_pred * contact_masks, contacts_batch)


        if steps_done % OUT_STEP == 0:
            print('Training, epoch {}, step: {}, loss_a: {}'.format(
                epoch, steps_done, loss_a))

        loss = loss_u + loss_a
        loss.backward()
        if steps_done % 30 == 0:
            all_optimizer.step()
            all_optimizer.zero_grad()
        steps_done = steps_done + 1

    if epoch % save_epi == 0:
        print('Validation set information')
        val_f1 = all_val(val_generator, ATTfold, device)
        print("val_tmp:", val_tmp)
        print("val_f1:", val_f1)
        if val_f1 > val_tmp:
            print("save the best model")
            val_tmp = val_f1
            torch.save(ATTfold.state_dict(), '../models_ckpt/ATTfold_d{}_s{}.pt'.format(d, CU_steps))
        else:
            print("save the low model")
            torch.save(ATTfold.state_dict(), '../models_ckpt/ATTfold_low_model.pt')

        print('-' * 100)

all_test(ATTfold, test_generator, test_data, device)



