import json
import numpy as np
import h5py
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import configs as cfgs

class CLEVRDataset(Dataset):
    def __init__(self, feature_h5, question_h5):
        """
        Image features are extracted according to:
        https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py
        """
        print('Loading data...')
        f1 = h5py.File(feature_h5, 'r')
        self.image_features = f1['features']
        f2 = h5py.File(question_h5, 'r')
        self.questions = f2['questions']
        self.img_idx = f2['image_idxs']
        self.answer = f2['answer']
        print('Done')

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[str(index)][:]
        answer = self.answer[index].tolist()
        img_feats = self.image_features[self.img_idx[index]]

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
        q_batch: torch tensor of shape (batch_size, padded_length, 300).
        a_batch: torch tensor of shape (batch_size, 1).
        lengths: list; valid length for each padded question.
    """
    # Sort a data list by question length (descending order).
    data.sort(key=lambda x: x[1].shape[0], reverse=True)
    img_feats, question, answer = zip(*data)

    # Merge images
    im_batch = torch.stack([torch.Tensor(feat).float() for feat in img_feats], 0)
    a_batch = torch.Tensor(answer)

    # Merge questions and add paddings
    lengths = [q.shape[0] for q in question]
    q_batch = torch.ones(len(question), max(lengths)) * cfgs.VOCAB_SIZE # +1 for <NULL> padding, -1 for 0 indexing
    for i, q in enumerate(question):
        q_batch[i, :lengths[i]] = torch.Tensor(q)
    return im_batch, q_batch, a_batch, lengths


train_dataset = CLEVRDataset(
    feature_h5=cfgs.TRAIN_IM_FEATS,
    question_h5=cfgs.TRAIN_QUESTION,
)

val_dataset = CLEVRDataset(
    feature_h5=cfgs.VAL_IM_FEATS,
    question_h5=cfgs.VAL_QUESTION,
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfgs.BATCH_SIZE, shuffle=True, collate_fn=collate_fn) #
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfgs.BATCH_SIZE, collate_fn=collate_fn)

# ## Testing
# btch = [train_dataset[i] for i in range(4)]
# im_batch, q_batch, a_batch, lengths = collate_fn(btch)
# print(q_batch)
# print(a_batch)
# print(lengths)
