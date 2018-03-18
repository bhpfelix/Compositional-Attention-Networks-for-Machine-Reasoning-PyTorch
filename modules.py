import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ControlUnit(nn.Module):

    def __init__(self, d):
        super(ControlUnit, self).__init__()
        self.d = d

        # W^{d,d}_i and b^d_i transform q into q_i (Section 3.2.1 Eq.(1))
        self.q_transform = nn.Linear(d, d)
        self.attn_weight = nn.Linear(d, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, transform, c_prev, q, cws):
        """
        Input:
        transform   -    shared W^{2d,d} and b^d that transforms [q_i, c_prev] into cq_i (Section 3.2.1 Eq.(2))
        c_prev      -    B x d        previous control state vector
        q           -    B x d        concatenation of the two final hidden states in biLSTM
        cws         -    B x d x S    contextual words from input unit, where S is the query length

        Return:
        c_i         -    B x d        control state vector
        """
        ## Question, if d is the output vector size of biLSTM, then the hidden state size of the biLSTM should be d/2 ?

        q_i = self.q_transform(q) # B x d
        cq_i = transform(torch.cat([q_i, c_prev], dim=1)) # B x d

        cv = cq_i.unsqueeze(2) * cws
        S = cv.size(2)
        print cv.size()
        print cv.view(-1, S).size()
        cv = self.attn_weight(cv.permute(0,2,1).contiguous().view(-1, self.d))
        cv = cv.view(-1, S)
        cv = self.softmax(cv).unsqueeze(1)

        c_i = cv * cws
        c_i = torch.sum(c_i, dim=2)

        return c_i

## Testing
B = 2 # batch size
d = 6 # dimension
S = 5 # seq length
lstm = nn.LSTM(
        input_size = 1,
        hidden_size = d // 2,
        num_layers = 1,
        bidirectional = True,
    )

inpt = Variable( torch.rand(S, B, 1) )
out, (h_t, c_t) = lstm(inpt)

# must call .contiguous() because the tensor is not a single block of memory, but a block with holes. view can be only used with contiguous tensors.
q = h_t.permute(1,0,2).contiguous().view(B, -1)
cws = out.permute(1,2,0)
transform = nn.Linear(2*d, d)
c_prev = Variable( torch.rand(B, d) )

model = ControlUnit(d)
c_i = model(transform, c_prev, q, cws)
print c_i.size()