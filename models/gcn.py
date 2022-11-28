import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init


class GCN(nn.Module):
    def __init__(self, nfeat=33, nhid1=16, nhid12=8, activation='LeakyRelu', mask_learning=True, nclass=1):
        super(GCN, self).__init__()
        # 给邻接矩阵加mask。初始化为1
        if mask_learning:
            inter_channels = 8  # 3. 双线性注意力
            self.conv_a = nn.Conv1d(nfeat, inter_channels, 1)
            self.conv_b = nn.Conv1d(nfeat, inter_channels, 1)
            self.inter_channels = inter_channels
            self.soft = nn.Softmax(dim=1)

        self.mask_learning = mask_learning

        self.gc1 = GraphConvolution_cat(nfeat, nhid1)
        self.gc2 = GraphConvolution_cat(nhid1, nhid12)
        # self.relu = nn.ReLU(inplace=True)  # nn.LeakyReLU(0.2)
        if activation =='Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax()
        elif activation == 'Relu':
            self.activation = nn.ReLU()
        elif activation == 'LeakyRelu':
            self.activation = nn.LeakyReLU()

        self.fc = nn.Sequential(
            nn.Linear(nhid12, nclass),
            nn.Sigmoid())

    def forward(self, x, adj):  # x:[20,50,33]  adj:[20,50,50]
        if self.mask_learning:
            # A1 = self.conv_a(x.transpose(1,2)).view(N, V, self.inter_channels) # 3. (conv的输入需要是NCV, x是NVC
            # A2 = self.conv_b(x.transpose(1,2)).view(N, self.inter_channels, V)
            A1 = self.conv_a(x.transpose(1,2)).transpose(1, 2)  # N V Cout
            A2 = self.conv_b(x.transpose(1,2)) # N Cout V
            A3 = self.soft(torch.matmul(A1, A2)) # N V V

            adj = torch.cat((A3, adj), 1) # N 2V V

        x = self.activation(self.gc1(x, adj))  # x:[20,50,16]
        x = self.activation(self.gc2(x, adj)) # [20,50,32]
        # 汇总节点特征表示整个图特征的过程  version2:尝试用Self-attention graph pooling
        x_pool = torch.max(x, dim=1).values   # x:[20,32]
        output = self.fc(x_pool)  # x:[20,1]
        return output


class GCN_noMetric(nn.Module):
    def __init__(self, nfeat=33, nhid1=16, nhid12=8, activation='LeakyRelu', mask_learning=True, nclass=1):
        super(GCN_noMetric, self).__init__()

        self.mask_learning = mask_learning

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid12)
        # self.relu = nn.ReLU(inplace=True)  # nn.LeakyReLU(0.2)
        if activation =='Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax()
        elif activation == 'Relu':
            self.activation = nn.ReLU()
        elif activation == 'LeakyRelu':
            self.activation = nn.LeakyReLU()
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(nhid12, nclass),
            nn.Sigmoid())

    def forward(self, x, adj):  # x:[20,50,33]  adj:[20,50,50]
        x = self.activation(self.gc1(x, adj))  # x:[20,50,16]
        x = self.activation(self.gc2(x, adj)) # [20,50,32]
        # 汇总节点特征表示整个图特征的过程
        x_pool = torch.max(x, dim=1).values   # x:[20,32]
        output = self.fc(x_pool)  # x:[20,1]
        return output


class GCN_noGaussian(nn.Module):
    def __init__(self, nfeat=33, nhid1=16, nhid12=8, activation='LeakyRelu', mask_learning=True, nclass=1):
        super(GCN_noGaussian, self).__init__()
        # 给邻接矩阵加mask。初始化为1
        if mask_learning:
            # self.mask = nn.Parameter(torch.ones(A_size)) # 1. 乘上mask
            # self.mask = nn.Parameter(torch.zeros(A_size)) # 2. 加上mask
            # nn.init.constant_(self.mask, 1e-6)
            inter_channels = 8  # 3. 双线性注意力
            self.conv_a = nn.Conv1d(nfeat, inter_channels, 1)
            self.conv_b = nn.Conv1d(nfeat, inter_channels, 1)
            self.inter_channels = inter_channels
            self.soft = nn.Softmax(dim=1)

        self.mask_learning = mask_learning

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid12)
        # self.relu = nn.ReLU(inplace=True)  # nn.LeakyReLU(0.2)
        if activation =='Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax()
        elif activation == 'Relu':
            self.activation = nn.ReLU()
        elif activation == 'LeakyRelu':
            self.activation = nn.LeakyReLU()
        # 全连接层
        # self.fc = nn.Linear(nhid12, nclass)
        self.fc = nn.Sequential(
            nn.Linear(nhid12, nclass),
            nn.Sigmoid())

    def forward(self, x, adj):  # x:[20,50,33]  adj:[20,50,50]
        N, V, C = x.size() # 20*50*33
        if self.mask_learning:
            # adj = adj*self.mask # 1.
            # adj = adj + self.mask # 2.
            A1 = self.conv_a(x.transpose(1, 2)).transpose(1, 2)  # N V Cout
            A2 = self.conv_b(x.transpose(1, 2))  # N Cout V
            A3 = self.soft(torch.matmul(A1, A2))  # N V V

            adj = A3  # N V V

        x = self.activation(self.gc1(x, adj))  # x:[20,50,16]
        x = self.activation(self.gc2(x, adj)) # [20,50,32]
        # 汇总节点特征表示整个图特征的过程
        x_pool = torch.max(x, dim=1).values   # x:[20,32]
        output = self.fc(x_pool)  # x:[20,1]
        return output


class baseline(nn.Module):
    def __init__(self, nfeat=33, nclass=1):
        super(baseline, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(nfeat, nclass),
            nn.Sigmoid())

    def forward(self, x, adj):  # x:[20,50,33]  adj:[20,50,50]
        # 汇总节点特征表示整个图特征的过程
        x_pool = torch.max(x, dim=1).values   # x:[20,32]
        output = self.fc(x_pool)  # x:[20,1]
        return output


# 只有一部分邻接矩阵（无拼接）的卷积层
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)  # 相乘 [20, 50, 33] [33 16]
        output = torch.einsum('nik, nkj -> nij', adj, support)  # [20, 50, 16] [20, 50, 50]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 有拼接邻接矩阵的卷积层
class GraphConvolution_cat(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_cat, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features*2, out_features)) # in_feature: C
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, adj):
        N, V, C = x.size()
        support = torch.matmul(adj, x)  # 相乘 [N V*2 V] [V C]
        support = support.view(N, V, 2*C)
        output = torch.einsum('nik, kj -> nij', support, self.weight)  # [N V 2*C] [2*C Cout]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
