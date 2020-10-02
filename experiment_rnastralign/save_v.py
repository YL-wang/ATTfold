import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch.optim as optim
from torch.utils import data

from ATTfold.data_generator import RNADataProcessor, Dataset
import collections
from ATTfold.ATTfold_model import ATTfold
from ATTfold.common.utils import *
from ATTfold.common.config import process_config
from ATTfold.evaluation import save_val

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
ATTfold_path = '../models_ckpt/ATTfold_0.809.pt'

epoches = config.EPOCHES
save_epi = config.evaluate_save
steps_done = 0
val_tmp = 0
LOAD_MODEL = True
#LOAD_MODEL = False

seed_torch()

RNA_SS_data = collections.namedtuple('RNA_SS_data',
                                     'seq ss_label length name pairs')

params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 6,
          'drop_last': False}
test_data = RNADataProcessor('../data_preprocess/{}/'.format(data_type), 'test_no_redundant_512')
test_set = Dataset(test_data)
test_generator = data.DataLoader(test_set, **params)


seq_len = test_data.data_y.shape[-2]
print('Max seq length ', seq_len)

ATTfold = ATTfold(d=d, L=seq_len,
                   steps = CU_steps).to(device)

if LOAD_MODEL and os.path.isfile(ATTfold_path):
    print('Loading ATTfold model...')
    ATTfold.load_state_dict(torch.load(ATTfold_path))


save_val(ATTfold, test_generator, test_data, device)
        






