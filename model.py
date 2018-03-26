import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

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
        KB_perm = KB.permute(0,2,3,1) # B x H x W x d
        _KB = KB_perm.contiguous().view(-1,self.d)
        _KB = self.KB_transform(_KB).view(B, H, W, d)
        I = _m_prev * _KB # B x H x W x d

        _I = torch.cat([I, KB_perm], dim=3).view(-1, 2*self.d)
        _I = self.merge_transform(_I).view(B, H, W, d)
        I = c_i.unsqueeze(1).unsqueeze(1) * _I

        mv = self.softmax(self.attn_weight(I.view(-1, self.d)).view(B, -1)).view(B, H, W, 1)

        m_new = mv * KB_perm
        m_new = torch.sum(m_new.view(B,-1,d), dim=1) # anchor on knowledge base
        return m_new


class WriteUnit(nn.Module):

    def __init__(self, d, merge_transform, self_attn=False, attn_weight=None, attn_merge=None, mem_gate=False, gate_transform=None):
        """
        Input:
        merge_transform    -    shared W^{2d,d} and b^d that transforms concatenated memory vec [m_new; m_prev] into m (Section 3.2.3 Eq.(8))

        self_attn          -    boolean flag for self attention variant
        attn_weight        -    shared W^{d,1} and b^1 that calculate attentional weights (Section 3.2.3 Eq.(9a))
        attn_merge         -    shared W^{2d,d} and b^d that transforms concatenated memory vec [m_sa; _m] into _m (Section 3.2.3 Eq.(10) if self_attn is True)

        mem_gate           -    boolean flag for memory gate variant
        gate_transform     -    shared W^{d,d} (W^{d,1} in paper?) and b^d that transforms c_i in to _c_i for gating purpose
        """
        super(WriteUnit, self).__init__()
        self.d = d
        self.merge_transform = merge_transform

        self.self_attn = self_attn
        self.attn_weight = attn_weight
        self.attn_merge = attn_merge

        self.mem_gate = mem_gate
        self.gate_transform = gate_transform

        self.softmax = nn.Softmax(dim=1)

    def forward(self, m_new, ls_m_i, c_i, ls_c_i):
        """
        Input:
        m_new       -    B x d                information vector retrieved by read unit
        ls_m_i      -    [B x d] x (i-1)      list of previous memory vectors, excluding current m_new
        c_i         -    B x d                current control vec
        ls_c_i      -    [B x d] x (i-1)      list of previous control vectors, excluding current c_i

        Return:
        ls_m_i      -    [B x d] x (i-1)      list of memory vectors, including current _m
        ls_c_i      -    [B x d] x (i-1)      list of control vectors, including current c_i
        """
        m_prev = ls_m_i[-1]
        _m = self.merge_transform(torch.cat([m_new, m_prev], dim=1))

        if self.self_attn:
            m_prevs = torch.stack(ls_m_i, dim=2)
            c_prevs = torch.stack(ls_c_i, dim=2)
            sa = c_i.unsqueeze(2) * c_prevs  # B x d x (i-1)
            sa = sa.permute(0,2,1).contiguous().view(-1, self.d)
            sa = self.softmax(self.attn_weight(sa).view(c_i.size(0), -1)).unsqueeze(1) # B x 1 x (i-1)
            m_sa = torch.sum(sa * m_prevs, dim=2)
            # incoporating new info with old info
            _m = self.attn_merge(torch.cat([m_sa, _m], dim=1))

        if self.mem_gate:
            _c_i = torch.sigmoid(self.gate_transform(c_i))
            _m = _c_i * m_prev + (1. - _c_i) * _m

        ls_c_i.append(c_i)
        ls_m_i.append(_m)

        return ls_c_i, ls_m_i


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

    def forward(self, ls_c_i, ls_m_i, q, cws, KB):
        """
        Input:
        ls_c_i      -    [B x d] x (i-1)  list of previous control vectors, --EXcluding-- current c_i
        ls_m_i      -    [B x d] x (i-1)  list of previous memory vectors, --EXcluding-- current _m
        q           -    B x d            concatenation of the two final hidden states in biLSTM
        cws         -    B x d x S        contextual words from input unit, where S is the query length
        KB          -    B x d x H x W    knowledge base (image feature map for VQA)

        Return:
        ls_c_i      -    [B x d] x i      list of previous control vectors, --INcluding-- current c_i
        _m          -    B x d            new memory state vector
        """
        c_prev = ls_c_i[-1]
        c_i = self.control_unit(c_prev, q, cws)

        m_prev = ls_m_i[-1]
        m_new = self.read_unit(m_prev, KB, c_i)

        ls_c_i, ls_m_i = self.write_unit(m_new, ls_m_i, c_i, ls_c_i)
        return ls_c_i, ls_m_i


class InputUnit(nn.Module):

    def __init__(self, d, vocab_size, embedding_dim, in_channels, hidden_channels=None):
        """
        Input:
        in_channels          -    image feature input channels
        hidden_channels      -    feature map channel after the first convolution layer
        """
        super(InputUnit, self).__init__()
        if d%2 == 1:
            raise ValueError('d needs to be an even integer')
        self.d = d
        self.char_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, d // 2, num_layers=1, bidirectional = True)

        if hidden_channels is None:
            hidden_channels = (in_channels + d) // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 2, padding=1) # Use 1 padding here to keep the dimension
        self.conv2 = nn.Conv2d(hidden_channels, d, 2)

        self.ac = nn.ELU()

    def forward(self, img_feats, questions, lengths):
        """
        Input:
        img_feats      -    B x (C+2*Pd) x H x W       image feature maps, where C is 1024 (ResNet101 conv4 output feature map number), and Pd is positional encoding dimension
        questions      -    B x S x (300+1)            question in 300 dimensional GloVe embedding, where S is the query length
        lengths        -    B                          list of B elements, holding length of input questions before padding

        Return:
        q           -    B x d            concatenation of the two final hidden states in biLSTM
        cws         -    B x d x S        contextual words from input unit, where S is the query length
        KB          -    B x d x H x W    knowledge base (image feature map for VQA)
        """
        B = questions.size(0)
        embeds = self.char_embeds(questions)
        packed_input = pack_padded_sequence(embeds.permute(1,0,2), lengths)
        packed_output, (h_t, c_t) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output)

        # must call .contiguous() because the tensor is not a single block of memory, but a block with holes. view can be only used with contiguous tensors.
        q = h_t.permute(1,0,2).contiguous().view(B, -1)
        cws = lstm_out.permute(1,2,0) # Figure 8

        KB = self.ac(self.conv1(img_feats))
        KB = self.ac(self.conv2(KB))

        return q, cws, KB


class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x


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
        # Double check where to apply dropout, also what kinds of dropout to apply
        # self.dropout = nn.Dropout(p=0.15)
        self.dropout = nn.AlphaDropout(p=0.15)
#         self.dropout = VariationalDropout(alpha=(0.15/0.85), dim=hidden_size)
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
        ans = self.fc2(ans) # output logits, since nn.CrossEntropy Takes care of softmax
        return ans


class CAN(nn.Module):

    def __init__(self, d, vocab_size, embedding_dim, input_in_channels, input_hidden_channels=None, p=12, write_self_attn=True, write_mem_gate=True, num_answers=28, out_hidden_size=None):
        """
        Compositional Attention Network

        Input:
        d  -
        p  -
        """
        super(CAN, self).__init__()
        self.d = d

        self.input_unit = InputUnit(d, vocab_size, embedding_dim, input_in_channels, input_hidden_channels)

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
        self.write_attn_merge = nn.Linear(2*d,d)
        self.write_mem_gate = write_mem_gate
        self.write_gate_transform = nn.Linear(d,1) # nn.Linear(d,d)
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
            ls_c_i = [Variable(torch.zeros(B,self.d)).cuda()]
            ls_m_i = [Variable(torch.zeros(B,self.d)).cuda()]
        else:
            ls_c_i = [Variable(torch.zeros(B,self.d))]
            ls_m_i = [Variable(torch.zeros(B,self.d))]
        return ls_c_i, ls_m_i

    def forward(self, img_feats, question, lengths):
        """
        Input:
        img_feats      -    B x (C+2*Pd) x H x W       image feature maps, where C is 1024 (ResNet101 conv4 output feature map number), and Pd is positional encoding dimension
        question       -    B x S x 300                question in 300 dimensional GloVe embedding, where S is the query length

        Return:
        ans         -    B x num_answers      softmax score over the answers
        """
        q, cws, KB = self.input_unit(img_feats, question, lengths)
        ls_c_i, ls_m_i = self._init_states(q.size(0))
        for mac in self.MACCells:
            ls_c_i, ls_m_i = mac(ls_c_i, ls_m_i, q, cws, KB)
        ans = self.output_unit(q, ls_m_i[-1])
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
