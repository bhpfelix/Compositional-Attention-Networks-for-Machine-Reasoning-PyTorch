import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ControlUnit(nn.Module):

    def __init__(self, d):
        super(ControlUnit, self).__init__()

        # W^{d,d}_i and b^d_i transform q into q_i (Section 3.2.1 Eq.(1))
        self.q_transform = nn.Linear(d, d)

    def forward(self, transform, c_prev, cws):
        """
        transform:       shared W^{2d,d} and b^d that transforms [q_i, c_prev] into cq_i (Section 3.2.1 Eq.(2))
        c_prev:          B x d        previous control vector
        cws:             B x d x S    contextual words from input unit, where S is the query length
        """
        q
        return x