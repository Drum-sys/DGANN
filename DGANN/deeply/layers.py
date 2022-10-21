import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionLayer(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    """
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数

        self.alpha = alpha     # leakyrelu激活的参数
   
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
        
        self.W_2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)  # xavier初始化
        
        self.W_3 = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W_3.data, gain=1.414)  # xavier初始化
        # self.a_2 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a_2.data, gain=1.414)   # xavier初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.leakyrelu2 = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)
        h_2 = torch.mm(inp, self.W_2)
        
        e = self.leakyrelu(torch.mm(h, h_2.t()))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_3 = torch.mm(inp, self.W_3)
        h_prime = torch.mm(attention, h_3)
          
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'