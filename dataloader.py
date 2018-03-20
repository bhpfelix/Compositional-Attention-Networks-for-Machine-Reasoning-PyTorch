import json
import numpy as np
import h5py
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import configs as cfgs

class CLEVRDataset(Dataset):
    def __init__(self, feature_h5, question_json, glove_json, answer_json, corpus, d_pos_vec):
        """
        Image features are extracted according to:
        https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py
        """
        print('Loading data...')
        f = h5py.File(feature_h5, 'r')
        self.image_features = f['features']
        qdata = json.load(open(question_json))
        self.questions = qdata['questions']
        self.glove = json.load(open(glove_json))
        self.answers = json.load(open(answer_json))
        self.corpus = corpus
        print('Done')

        # Paper did not report this hyperparameter, need to look into this
        self.d_pos_vec = d_pos_vec
        self.pos_code = None

    def __len__(self):
        return len(self.questions)

    def question_to_embedding(self, q):
        # According to FiLM, not including "?" sign, but include ";"
        _q = q.lower().replace('?','').replace(';',' ;') # pad space for split
        return np.array([self.glove[w][self.corpus] for w in _q.split()])

    def position_encoding(self, n_position):
        '''
        Init the sinusoid position encoding table
        https://github.com/jadore801120/attention-is-all-you-need-pytorch
        '''
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / self.d_pos_vec) for j in range(self.d_pos_vec)]
            if pos != 0 else np.zeros(self.d_pos_vec) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return position_enc

    def pos_encode_block(self, h, w):
        hcode = self.position_encoding(h) # h x d
        wcode = self.position_encoding(w) # w x d

        hcode = np.transpose(np.concatenate([np.expand_dims(hcode, axis=1)]*w, axis=1), (2,0,1))
        wcode = np.transpose(np.concatenate([np.expand_dims(wcode, axis=0)]*h, axis=0), (2,0,1))

        return np.concatenate([hcode, wcode], axis=0)

    def __getitem__(self, index):
        q = self.questions[index]
        question = self.question_to_embedding(q['question'])
        answer = self.answers.index(q['answer'])
        img_feats = self.image_features[q['image_index']]
        if self.pos_code is None:
            _, H, W = img_feats.shape
            self.pos_code = self.pos_encode_block(H, W)
        img_feats = np.concatenate([img_feats, self.pos_code], axis=0)

        return img_feats, question, answer

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (img_feats, question, answer).

    We should build custom collate_fn rather than using default collate_fn,
    because merging question (including padding) is not supported in default.
    Args:
        data: list of tuple (img_feats, question, answer).
            - img_feats: torch tensor of shape (C, H, W).
            - question: torch tensor of shape (S, 300); variable length S.
            - answer: int
    Returns:
        im_batch: torch tensor of shape (batch_size, C, H, W).
        q_batch: torch tensor of shape (batch_size, padded_length, 300+1).
        a_batch: torch tensor of shape (batch_size, 1).
        lengths: list; valid length for each padded question.
    """
    # Sort a data list by question length (descending order).
    data.sort(key=lambda x: x[1].shape[0], reverse=True)
    img_feats, question, answer = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    im_batch = torch.stack([torch.Tensor(feat).float() for feat in img_feats], 0)
    a_batch = torch.Tensor(answer)

    # Merge captions (from tuple of 2D tensor to 3D tensor).
    lengths = [q.shape[0] for q in question]
    q_batch = torch.zeros(len(question), max(lengths), question[0].shape[1]+1).float() # +1 for padding channel
    for i, q in enumerate(question):
        q_batch[i, :lengths[i], :-1] = torch.Tensor(q).float()
        if lengths[i] < max(lengths):
            q_batch[i, lengths[i]:, -1] = 1.
    return im_batch, q_batch, a_batch # , lengths

train_dataset = CLEVRDataset(
    feature_h5=cfgs.TRAIN_IM_FEATS,
    question_json=cfgs.TRAIN_QUESTION,
    glove_json=cfgs.GLOVE_PATH,
    answer_json=cfgs.ANSWER_PATH,
    corpus=cfgs.CORPUS,
    d_pos_vec=cfgs.D_POS_VEC
)

val_dataset = CLEVRDataset(
    feature_h5=cfgs.VAL_IM_FEATS,
    question_json=cfgs.VAL_QUESTION,
    glove_json=cfgs.GLOVE_PATH,
    answer_json=cfgs.ANSWER_PATH,
    corpus=cfgs.CORPUS,
    d_pos_vec=cfgs.D_POS_VEC
)

## Example Usage
# img_feats, question, answer = dataset[0]
# print(img_feats.shape)
# print(question.shape)
# print(answer)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfgs.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfgs.BATCH_SIZE, collate_fn=collate_fn)

