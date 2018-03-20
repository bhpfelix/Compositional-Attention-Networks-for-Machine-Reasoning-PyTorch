# build dataset from raw CLEVR
"""
1. get conv4 features of the images from ResNet101 pretrained on ImageNet
2. get word embeddings of queries
3. save instance

Question:
1. During question processing, include question mark as vocab? doesn't make sense because all questions have question mark
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

__author__ = 'Haoping Bai'
__copyright__ = 'Copyright (c) 2018, Haoping Bai'
__email__ = 'bhpfelix@gmail.com'
__license__ = 'MIT'

def build_dataset(CLEVR_root, glove_dict_path):

    return None


model = torchvision.models.resnet101(pretrained=True)