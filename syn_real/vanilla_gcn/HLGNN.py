import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, TransformerConv, GATConv
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul


class HLGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout, alpha, init):
        super(HLGNN, self).__init__(aggr='add')
        self.K = K
        self.init = init
        self.alpha = alpha
        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)

        assert init in ['SGC', 'RWR', 'KI', 'Random']
        if init == 'SGC':
            alpha = int(alpha)
            TEMP = 0.0 * np.ones(K+1)
            TEMP[alpha] = 1.0
        elif init == 'RWR':
            TEMP = alpha * (1-alpha) ** np.arange(K+1)
            TEMP[-1] = (1-alpha) ** K
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif init == 'KI':
            TEMP = alpha ** np.arange(K+1)
        elif init == 'Random':
            bound = np.sqrt(3 / (K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP / np.sum(np.abs(TEMP))

        self.temp = Parameter(torch.tensor(TEMP))
        # self.beta = Parameter(torch.zeros(3))
        
    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.init == 'SGC':
            self.alpha = int(self.alpha)
            self.temp.data[self.alpha]= 1.0
        elif self.init == 'RWR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha * (1-self.alpha) ** k
            self.temp.data[-1] = (1-self.alpha) ** self.K
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.init == 'KI':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha ** k
            # self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.init == 'Random':
            bound = np.sqrt(3 / (self.K+1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))


    def forward(self, x, adj_t, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        adj_t = gcn_norm(adj_t, edge_weight, adj_t.size(0), dtype=torch.float)
        
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(adj_t, x=x, edge_weight=edge_weight, size=None)
            gamma = self.temp[k+1]
            hidden = hidden + gamma * x
        return hidden
    
    
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce="add")