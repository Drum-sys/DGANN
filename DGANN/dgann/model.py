import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution,GraphAttentionLayer
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class Encoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(Encoder, self).__init__()

        self.dropout = dropout

        self.encgc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout=dropout)
        self.encgc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout=dropout)

        self.decgc1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout=dropout)
        self.decgc2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout=dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.encgc1(x, adj)
        hidden2 = self.encgc2(hidden1, adj)
        return hidden2

    def decode(self, x, adj):
        return self.decgc2(self.decgc1(x, adj), adj)

    def forward(self, x, adj):
        enc = self.encode(x, adj)
        dec = self.decode(enc, adj)
        return dec, enc


class GAT(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, alpha, dropout):
        super(GAT, self).__init__()
#         self.encgc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
#         self.encgc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.encgc1 = GraphAttentionLayer(input_feat_dim, hidden_dim1, alpha, dropout)
#         self.encgc2 = GraphAttentionLayer(hidden_dim1,hidden_dim2, alpha, dropout)

#         self.decgc1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
#         self.decgc2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=lambda x: x)
        self.decgc1 = GraphAttentionLayer(hidden_dim1, input_feat_dim, alpha, dropout)
#         self.decgc2 = GraphAttentionLayer(hidden_dim1, input_feat_dim, alpha, dropout)

    def encode(self, x, adj):
        hidden1 = self.encgc1(x, adj)
#         hidden2 = self.encgc2(hidden1, adj)
        return hidden1

    def decode(self, hidden, adj):
#         hidden1 = self.decgc1(hidden, adj)
        recon = self.decgc1(hidden, adj)
        return recon

    def forward(self, x, adj):
        enc = self.encode(x, adj)
        dec = self.decode(enc, adj)
        return dec, enc

class SelfExpr(Module):
    def __init__(self, n):
        self.n = n
        super(SelfExpr, self).__init__()
        #self.weight = Parameter(1*torch.FloatTensor(n, n))
        self.weight = Parameter(torch.FloatTensor(n, n).uniform_(0,0.01))

    def forward(self, input):
        #self.weight.data = F.relu(self.weight)
        output = torch.mm(self.weight-torch.diag(torch.diagonal(self.weight)), input)
        return self.weight, output
    
    def reset(self, input):
        self.weight.data = torch.FloatTensor(self.n, self.n).uniform_(0,0.01)
        

class ClusterModel(torch.nn.Module):
    def __init__(self, n_hid1, n_hid2, n_class, dropout):
        super(ClusterModel, self).__init__()
        self.mlp1 = torch.nn.Linear(n_hid1, n_hid2)
        self.mlp2 = torch.nn.Linear(n_hid2, n_class)
        self.dropout = dropout
        # ~ torch.nn.init.xavier_uniform_(self.mlp1.weight)
        # ~ torch.nn.init.xavier_uniform_(self.mlp2.weight)
    
    def forward(self, x1: torch.Tensor) -> torch.Tensor:
            x2 = F.relu(self.mlp1(x1))
            # if self.dropout > 0:
            #     x2 = F.dropout(x2, self.dropout)
            z = F.softmax(self.mlp2(x2), dim=-1)
            #z = F.relu(self.fc4(x3))
            return z