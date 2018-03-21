import torch
USE_CUDA = torch.cuda.is_available()

GLOVE_PATH = 'glove.json'
ANSWER_PATH = 'answers.json'
CORPUS = '42B' # '840B'
D_POS_VEC = 16 # 128
TRAIN_IM_FEATS = 'train.hdf5'
TRAIN_QUESTION = '/disk2/CLEVR_v1.0/questions/CLEVR_train_questions.json'
VAL_IM_FEATS = 'val.hdf5'
VAL_QUESTION = '/disk2/CLEVR_v1.0/questions/CLEVR_val_questions.json'

BATCH_SIZE = 64 #
NET_PARAM = {
    'd':512,
    'input_in_channels':1024 + 2 * D_POS_VEC,
    'input_hidden_channels':None,
    'p':1,
    'write_self_attn':True, # False, #
    'write_mem_gate':True,
    'num_answers':28,
    'out_hidden_size':512
}

OPT_PARAM = {
    'lr':0.0001,
    'betas':(0.9, 0.999),
    'eps':1e-08,
    'weight_decay':0 # 0.999
}