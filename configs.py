import torch
USE_CUDA = torch.cuda.is_available()

GLOVE_PATH = 'glove.json'
ANSWER_PATH = 'answers.json'
TRAIN_IM_FEATS = 'train.hdf5'
TRAIN_QUESTION = 'train_questions.hdf5'
VAL_IM_FEATS = 'val.hdf5'
VAL_QUESTION = 'val_questions.hdf5'

VOCAB_SIZE = 80 + 1 # +1 to include ";"
BATCH_SIZE = 64 #
NET_PARAM = {
    'd':512,
    'vocab_size':VOCAB_SIZE + 1, # +1 to include <NULL> appended during preprocessing,
    'embedding_dim':300,
    'input_in_channels':1024,
    'input_hidden_channels':None,
    'p':12,
    'write_self_attn':True,
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
