import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class GCN(nn.Module):
    def __init__(self, opt, pos_encoding, nfeat, nhid, out_dim, dropout, device):
        super(GCN, self).__init__()

        self.opt = opt
        self.pos_encoding = pos_encoding
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out_dim)
        self.dropout = dropout
        self.device = device
        
        if opt['beltrami']:
            self.mx = nn.Linear(nfeat, opt['feat_hidden_dim'])
            self.mp = nn.Linear(opt['pos_enc_dim'], opt['pos_enc_hidden_dim'])
            opt['hidden_dim'] = opt['feat_hidden_dim'] + opt['pos_enc_hidden_dim']

    def forward(self, x, adj):
        if self.opt['beltrami']:
            pos_encoding = self.pos_encoding.to(self.device) if self.pos_encoding is not None else None
            
            x = F.dropout(x, self.opt['input_dropout'], training=self.training)
            x = self.mx(x)
            p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
            p = self.mp(p)
            x = torch.cat([x, p], dim=1)
            
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x