import shutil, os, csv, itertools, glob

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import configs as cfgs

from modules import CAN
from dataloader import train_loader, val_loader

CAN(**cfgs.NET_PARAM)
