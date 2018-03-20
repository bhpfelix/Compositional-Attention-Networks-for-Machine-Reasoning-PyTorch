import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import configs as cfgs

__author__ = 'Haoping Bai'
__copyright__ = 'Copyright (c) 2018, Haoping Bai'
__email__ = 'bhpfelix@gmail.com'
__license__ = 'MIT'

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
        merge_transform    -    shared W^{2d,d} and b^d that transforms concatenated hypercolumn [I_hw; KB_hw] into I'_hw (Section 3.2.2 Eq.(6a))
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

# ## Testing
# B = 2
# d = 6
# H, W = 14, 14
# m_prev_transform = nn.Linear(d,d)
# KB_transform = nn.Linear(d,d)
# merge_transform = nn.Linear(2*d,d)
# attn_weight = nn.Linear(d,1)

# model = ReadUnit(d, m_prev_transform, KB_transform, merge_transform, attn_weight)

# m_prev = Variable(torch.rand(B,d))
# KB = Variable(torch.rand(B,d,H,W))
# c_i = Variable(torch.rand(B,d))

# m_new = model(m_prev, KB, c_i)
# print m_new.size()

class WriteUnit(nn.Module):

    def __init__(self, d, merge_transform, self_attn=False, attn_weight=None, attn_merge=None, mem_gate=False, gate_transform=None):
        """
        Input:
        merge_transform    -    shared W^{2d,d} and b^d that transforms concatenated memory vec [m_new; m_prev] into m (Section 3.2.3 Eq.(8))

        self_attn          -    boolean flag for self attention variant
        attn_weight        -    shared W^{d,1} and b^1 that calculate attentional weights (Section 3.2.3 Eq.(9a))
        attn_merge         -    shared W^{2d,d} and b^d that transforms concatenated memory vec [m_sa; m] into _m (Section 3.2.3 Eq.(10))

        mem_gate           -    boolean flag for memory gate variant
        gate_transform     -    shared W^{d,d} and b^d that transforms c_i in to _c_i for gating purpose
        """
        super(WriteUnit, self).__init__()
        self.d = d
        self.merge_transform = merge_transform

        self.self_attn = self_attn
        self.attn_weight = attn_weight
        self.attn_merge = attn_merge

        self.mem_gate = mem_gate
        self.gate_transform = gate_transform

    def forward(self, m_new, m_prev, ls_c_i=None):
        """
        Input:
        m_new       -    B x d            information vector retrieved by read unit
        m_prev      -    B x d            previous memory state vector
        ls_c_i      -    [B x d] x i      list of previous control vectors, including current c_i

        Return:
        _m          -    B x d            new memory state vector
        """
        _m = self.merge_transform(torch.cat([m_new, m_prev], dim=1))

        if self.self_attn:
            c_i = ls_c_i[-1].unsqueeze(2)
            c_prevs = torch.stack(ls_c_i[:-1], dim=2)
            sa = c_i * c_prevs  # B x d x (i-1)
            sa = sa.permute(0,2,1).contiguous().view(-1, self.d)
            sa = self.attn_weight(sa).view(c_i.size(0), -1).unsqueeze(1) # B x 1 x (i-1)
            m_sa = torch.sum(sa * c_prevs, dim=2)

            _m = self.attn_merge(torch.cat([m_sa, _m], dim=1))

        if self.mem_gate:
            c_i = ls_c_i[-1]
            _c_i = torch.sigmoid(self.gate_transform(c_i))
            _m = _c_i * m_prev + (1. - _c_i) * _m

        return _m

# ## Testing
# B = 2
# d = 6
# merge_transform = nn.Linear(2*d,d)

# self_attn = True # False #
# attn_weight = nn.Linear(d,1)
# attn_merge = nn.Linear(2*d, d)

# mem_gate = True # False #
# gate_transform = nn.Linear(d,d)

# model = WriteUnit(d, merge_transform, self_attn, attn_weight, attn_merge, mem_gate, gate_transform)

# m_new = Variable(torch.rand(B,d))
# m_prev = Variable(torch.rand(B,d))
# ls_c_i = [Variable(torch.rand(B,d)) for _ in range(10)]

# m = model(m_new, m_prev, ls_c_i)
# print m.size()


class MACCell(nn.Module):

    def __init__(self, d, ctrl_params, read_params, write_params):
        """
        Memory, Attention, and Control (MAC) cell

        Input:
        self_attn          -    boolean flag for self attention variant
        mem_gate           -    boolean flag for memory gate variant
        """
        super(MACCell, self).__init__()
        self.d = d

        # Initialize units
        self.control_unit = ControlUnit(**ctrl_params)
        self.read_unit = ReadUnit(**read_params)
        self.write_unit = WriteUnit(**write_params)

    def forward(self, ls_c_i, m_prev, q, cws, KB):
        """
        Input:
        ls_c_i      -    [B x d] x (i-1)  list of previous control vectors, --EXcluding-- current c_i
        m_prev      -    B x d            previous memory state vector
        q           -    B x d            concatenation of the two final hidden states in biLSTM
        cws         -    B x d x S        contextual words from input unit, where S is the query length
        KB          -    B x d x H x W    knowledge base (image feature map for VQA)

        Return:
        ls_c_i      -    [B x d] x i      list of previous control vectors, --INcluding-- current c_i
        _m          -    B x d            new memory state vector
        """
        c_prev = ls_c_i[-1]
        c_i = self.control_unit(c_prev, q, cws)
        ls_c_i.append(c_i)

        m_new = self.read_unit(m_prev, KB, c_i)

        _m = self.write_unit(m_new, m_prev, ls_c_i)
        return ls_c_i, _m

# ## Testing
# B = 2 # batch size
# d = 6 # dimension
# S = 5 # seq length
# H, W = 14, 14

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

# ls_c_i = [Variable(torch.rand(B,d)) for _ in range(10)]
# m_prev = Variable(torch.rand(B,d))

# KB = Variable(torch.rand(B,d,H,W))

# ctrl_params = {
#     'd':d,
#     'transform':nn.Linear(2*d, d),
#     'attn_weight':nn.Linear(d, 1),
# }

# read_params = {
#     'd':d,
#     'm_prev_transform':nn.Linear(d,d),
#     'KB_transform':nn.Linear(d,d),
#     'merge_transform':nn.Linear(2*d,d),
#     'attn_weight':nn.Linear(d,1),
# }

# write_params = {
#     'd':d,
#     'merge_transform':nn.Linear(2*d,d),
#     'self_attn':True,
#     'attn_weight':nn.Linear(d,1),
#     'attn_merge':nn.Linear(2*d, d),
#     'mem_gate':True,
#     'gate_transform':nn.Linear(d,d),
# }

# model = MACCell(d, ctrl_params, read_params, write_params)
# ls_c_i, _m = model(ls_c_i, m_prev, q, cws, KB)

# print len(ls_c_i)
# for item in ls_c_i:
#     print item.size()
# print _m.size()


class InputUnit(nn.Module):

    def __init__(self, d, in_channels, hidden_channels=None):
        """
        Input:
        in_channels          -    image feature input channels
        hidden_channels      -    feature map channel after the first convolution layer
        """
        super(InputUnit, self).__init__()
        if d%2 == 1:
            raise ValueError('d needs to be an even integer')
        self.d = d
        self.lstm = nn.LSTM(
                input_size = 300 + 1, # GloVe dimension + padding channel (Double check how to handdle variable length input)
                hidden_size = d // 2,
                num_layers = 1,
                bidirectional = True,
            )
        if hidden_channels is None:
            hidden_channels = (in_channels + d) // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 2, padding=1) # Use 1 padding here to keep the dimension
        self.conv2 = nn.Conv2d(hidden_channels, d, 2)

        self.ac = nn.ELU()

    def forward(self, img_feats, question):
        """
        Input:
        img_feats      -    B x (C+2*Pd) x H x W       image feature maps, where C is 1024 (ResNet101 conv4 output feature map number), and Pd is positional encoding dimension
        question       -    B x S x (300+1)                question in 300 dimensional GloVe embedding, where S is the query length


        Return:
        q           -    B x d            concatenation of the two final hidden states in biLSTM
        cws         -    B x d x S        contextual words from input unit, where S is the query length
        KB          -    B x d x H x W    knowledge base (image feature map for VQA)
        """
        B = question.size(0)
        out, (h_t, c_t) = self.lstm(question.permute(1,0,2))
        # must call .contiguous() because the tensor is not a single block of memory, but a block with holes. view can be only used with contiguous tensors.
        q = h_t.permute(1,0,2).contiguous().view(B, -1)
        cws = out.permute(1,2,0)

        KB = self.ac(self.conv1(img_feats))
        KB = self.ac(self.conv2(KB))

        return q, cws, KB

# ## Testing
# B = 2
# d = 6
# H, W = 14, 14
# S = 5
# in_channels = 1280
# hidden_channels = (in_channels + d) // 2

# img_feats = Variable( torch.rand(B, in_channels, H, W) )
# question = Variable( torch.rand(B, S, 300) )

# model = InputUnit(d, in_channels, hidden_channels)
# q, cws, KB = model(img_feats, question)
# print q.size()
# print cws.size()
# print KB.size()


class OutputUnit(nn.Module):

    def __init__(self, d, num_answers=28, hidden_size=None):
        """
        Input:
        num_answers          -    number of all possible answers, for CLEVR, is 28
        """
        super(OutputUnit, self).__init__()
        if hidden_size is None:
            hidden_size = (2*d + num_answers) // 2
        self.fc1 = nn.Linear(2*d, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_answers)

        self.ac = nn.ELU()
        # Double check where to apply dropout, the paper did not state
        self.dropout = nn.Dropout(p=0.85)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, mp):
        """
        Input:
        q           -    B x d                question repr, concatenation of the two final hidden states in biLSTM
        mp          -    B x d                memory state vector from the last MAC Cell

        Return:
        ans         -    B x num_answers      softmax score over the answers
        """
        ans = torch.cat([q, mp], dim=1)
        ans = self.dropout(self.ac(self.fc1(ans)))
        ans = self.softmax(self.ac(self.fc2(ans)))
        return ans

# ## Testing
# B = 2
# d = 6

# q = Variable(torch.rand(B,d))
# mp = Variable(torch.rand(B,d))

# model = OutputUnit(d)
# ans = model(q, mp)
# print ans.size()


class CAN(nn.Module):

    def __init__(self, d, input_in_channels, input_hidden_channels=None, p=12, write_self_attn=True, write_mem_gate=True, num_answers=28, out_hidden_size=None):
        """
        Compositional Attention Network

        Input:
        d  -
        p  -
        """
        super(CAN, self).__init__()
        self.d = d

        self.input_unit = InputUnit(d, input_in_channels, input_hidden_channels)

        self.ctrl_transform = nn.Linear(2*d, d)
        self.ctrl_attn_weight = nn.Linear(d, 1)
        ctrl_params = {
            'd':d,
            'transform':self.ctrl_transform,
            'attn_weight':self.ctrl_attn_weight,
        }

        self.read_m_prev_transform = nn.Linear(d,d)
        self.read_KB_transform = nn.Linear(d,d)
        self.read_merge_transform = nn.Linear(2*d,d)
        self.read_attn_weight = nn.Linear(d,1)
        read_params = {
            'd':d,
            'm_prev_transform':self.read_m_prev_transform,
            'KB_transform':self.read_KB_transform,
            'merge_transform':self.read_merge_transform,
            'attn_weight':self.read_attn_weight,
        }

        self.write_merge_transform = nn.Linear(2*d,d)
        self.write_self_attn = write_self_attn
        self.write_attn_weight = nn.Linear(d,1)
        self.write_attn_merge = nn.Linear(2*d, d)
        self.write_mem_gate = write_mem_gate
        self.write_gate_transform = nn.Linear(d,d)
        write_params = {
            'd':d,
            'merge_transform':self.write_merge_transform,
            'self_attn':self.write_self_attn,
            'attn_weight':self.write_attn_weight,
            'attn_merge':self.write_attn_merge,
            'mem_gate':self.write_mem_gate,
            'gate_transform':self.write_gate_transform,
        }

        self.MACCells = nn.ModuleList([MACCell(d, ctrl_params, read_params, write_params) for _ in range(p)])

        self.output_unit = OutputUnit(d, num_answers, out_hidden_size)

    def _init_states(self, B):
        if cfgs.USE_CUDA:
            ls_c_i = [Variable(torch.rand(B,self.d)).cuda()]
            _m = Variable(torch.rand(B,self.d)).cuda()
        else:
            ls_c_i = [Variable(torch.rand(B,self.d))]
            _m = Variable(torch.rand(B,self.d))
        return ls_c_i, _m

    def forward(self, img_feats, question):
        """
        Input:
        img_feats      -    B x (C+2*Pd) x H x W       image feature maps, where C is 1024 (ResNet101 conv4 output feature map number), and Pd is positional encoding dimension
        question       -    B x S x 300                question in 300 dimensional GloVe embedding, where S is the query length

        Return:
        ans         -    B x num_answers      softmax score over the answers
        """
        q, cws, KB = self.input_unit(img_feats, question)
        ls_c_i, _m = self._init_states(q.size(0))
        for mac in self.MACCells:
            ls_c_i, _m = mac(ls_c_i, _m, q, cws, KB)
        ans = self.output_unit(q, _m)
        return ans

# ## Testing
# B = 4
# d = 60
# H, W = 14, 14
# S = 5
# in_channels = 1280

# img_feats = Variable( torch.rand(B, in_channels, H, W) )
# question = Variable( torch.rand(B, S, 300) )

# model = CAN(d, input_in_channels=in_channels, input_hidden_channels=None, p=12, write_self_attn=True, write_mem_gate=True, num_answers=28, out_hidden_size=None)
# ans = model(img_feats, question)
# print ans.size()


# ## initialize model
# def get_can(resume=False, resume_path=None):
#     model = CAN(**cfgs.NET_PARAM)
#     if cfgs.USE_CUDA:
#         model.cuda()
#     return model