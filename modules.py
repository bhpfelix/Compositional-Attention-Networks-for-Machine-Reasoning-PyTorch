import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ControlUnit(nn.Module):

    def __init__(self, d, transform, attn_weight):
        """
        transform   -    shared W^{2d,d} and b^d that transforms [q_i, c_prev] into cq_i (Section 3.2.1 Eq.(2))
        attn_weight -    shared W^{d,1} and b^1 that calculate attentional weights (Section 3.2.1 Eq.(3a))
        """
        super(ControlUnit, self).__init__()
        self.d = d
        self.attn_weight = attn_weight
        self.transform = transform

        # W^{d,d}_i and b^d_i transform q into q_i (Section 3.2.1 Eq.(1))
        # Unlike all other params which are shared across cells, this is unique to each cell
        self.q_transform = nn.Linear(d, d)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, c_prev, q, cws):
        """
        Input:
        c_prev      -    B x d        previous control state vector
        q           -    B x d        concatenation of the two final hidden states in biLSTM
        cws         -    B x d x S    contextual words from input unit, where S is the query length

        Return:
        c_i         -    B x d        control state vector
        """
        ## Question, if d is the output vector size of biLSTM, then the hidden state size of the biLSTM should be d/2 ?

        q_i = self.q_transform(q) # B x d
        cq_i = self.transform(torch.cat([q_i, c_prev], dim=1)) # B x d

        cv = cq_i.unsqueeze(2) * cws
        S = cv.size(2)

        # Note: look into the efficiency of .contiguous()
        # must call .contiguous() because the tensor is not a single block of memory, but a block with holes. view can be only used with contiguous tensors.
        cv = self.attn_weight(cv.permute(0,2,1).contiguous().view(-1, self.d))
        cv = cv.view(-1, S)
        cv = self.softmax(cv).unsqueeze(1)

        c_i = cv * cws
        c_i = torch.sum(c_i, dim=2)

        return c_i

# ## Testing
# B = 2 # batch size
# d = 6 # dimension
# S = 5 # seq length
# lstm = nn.LSTM(
#         input_size = 1,
#         hidden_size = d // 2,
#         num_layers = 1,
#         bidirectional = True,
#     )

# inpt = Variable( torch.rand(S, B, 1) )
# out, (h_t, c_t) = lstm(inpt)

# # must call .contiguous() because the tensor is not a single block of memory, but a block with holes. view can be only used with contiguous tensors.
# q = h_t.permute(1,0,2).contiguous().view(B, -1)
# cws = out.permute(1,2,0)
# transform = nn.Linear(2*d, d)
# attn_weight = nn.Linear(d, 1)
# c_prev = Variable( torch.rand(B, d) )

# model = ControlUnit(d, transform, attn_weight)
# c_i = model(c_prev, q, cws)
# print c_i.size()


class ReadUnit(nn.Module):

    def __init__(self, d, m_prev_transform, KB_transform, merge_transform, attn_weight):
        """
        Input:
        m_prev_transform   -    shared W^{d,d} and b^d that transforms m_prev into m'_prev (Section 3.2.2 Eq.(4))
        KB_transform       -    shared W^{d,d} and b^d that transforms each hypercolumn KB_hw into KB'_hw (Section 3.2.2 Eq.(5a))
        merge_transform    -    shared W^{2d,d} and b^d that transforms merged hypercolumn [I_hw; KB_hw] into I'_hw (Section 3.2.2 Eq.(6a))
        attn_weight        -    shared W^{d,1} and b^1 that calculate attentional weights (Section 3.2.2 Eq.(7a))
        """
        super(ReadUnit, self).__init__()
        self.d = d
        self.m_prev_transform = m_prev_transform
        self.KB_transform = KB_transform
        self.merge_transform = merge_transform

        self.attn_weight = attn_weight

        self.softmax = nn.Softmax(dim=1)

    def forward(self, m_prev, KB, c_i):
        """
        Input:
        m_prev      -    B x d            previous memory state vector
        KB          -    B x d x H x W    knowledge base (image feature map for VQA)
        c_i         -    B x d            control state vector

        Return:
        m_new       -    B x d            retrieved information vector
        """

        _m_prev = self.m_prev_transform(m_prev).unsqueeze(1).unsqueeze(1)
        B, d, H, W = KB.size()
        KB = KB.permute(0,2,3,1) # B x H x W x d
        _KB = KB.contiguous().view(-1,self.d)
        _KB = self.KB_transform(_KB).view(B, H, W, d)
        I = _m_prev * _KB # B x H x W x d

        _I = torch.cat([I, KB], dim=3).view(-1, 2*self.d)
        _I = self.merge_transform(_I).view(B, H, W, d)
        I = c_i.unsqueeze(1).unsqueeze(1) * _I

        mv = self.attn_weight(I.view(-1, self.d)).view(B, H, W, 1)

        m_new = mv * KB
        m_new = torch.sum(m_new.view(B,-1,d), dim=1) # anchor on knowledge base
        return m_new

## Testing
B = 2
d = 6
H, W = 14, 14
m_prev_transform = nn.Linear(d,d)
KB_transform = nn.Linear(d,d)
merge_transform = nn.Linear(2*d,d)
attn_weight = nn.Linear(d,1)

model = ReadUnit(d, m_prev_transform, KB_transform, merge_transform, attn_weight)

m_prev = Variable(torch.rand(B,d))
KB = Variable(torch.rand(B,d,H,W))
c_i = Variable(torch.rand(B,d))

m_new = model(m_prev, KB, c_i)
print m_new.size()