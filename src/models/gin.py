import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool

from src.models.base import BaseModel


num_atom_type = 119
num_chirality_tag = 4


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.linear_or_not = True
        self.num_layers = num_layers
        
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GINConv(MessagePassing):
    def __init__(self, emb_dim, num_mlp_layers=2, learn_eps=True, neighbor_pooling_type="sum"):
        super(GINConv, self).__init__(aggr="add")
        
        self.emb_dim = emb_dim
        self.learn_eps = learn_eps
        self.neighbor_pooling_type = neighbor_pooling_type
        
        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        
        self.mlp = MLP(num_mlp_layers, emb_dim, emb_dim, emb_dim)
    
    def forward(self, x, edge_index):
        if self.learn_eps:
            out = self.propagate(edge_index, x=x)
            out = (1 + self.eps) * x + out
        else:
            out = self.propagate(edge_index, x=x)
            out = out + x
        
        out = self.mlp(out)
        return out
    
    def message(self, x_j):
        return x_j


class GIN(BaseModel):
    def __init__(
        self,
        in_feat: int = 113,
        hidden_feat: int = 64,
        out_feat: int = 32,
        out: int = 1,
        grid_feat: int = 1,
        num_layers: int = 5,
        pooling: str = "mean",
        use_bias: bool = False,
        drop_ratio: float = 0.5,
        feat_dim: int = 256,
        num_mlp_layers: int = 2,
        learn_eps: bool = True,
        neighbor_pooling_type: str = "sum",
        JK: str = "concat",
    ):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = hidden_feat * 2
        self.pooling = pooling
        self.learn_eps = learn_eps
        self.neighbor_pooling_type = neighbor_pooling_type
        
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.gnns.append(
                GINConv(
                    self.emb_dim,
                    num_mlp_layers=num_mlp_layers,
                    learn_eps=learn_eps,
                    neighbor_pooling_type=neighbor_pooling_type,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))
        
        if pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool
        
        if self.JK == "concat":
            self.feat_lin = nn.Linear(self.emb_dim * (num_layers + 1), feat_dim)
        else:
            self.feat_lin = nn.Linear(self.emb_dim, feat_dim)
        
        self.pred_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, out),
        )
    
    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        
        h_list = [x]
        for layer in range(self.num_layers):
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=-1), dim=-1)[0]
        else:
            node_representation = h_list[-1]
            for h in h_list[:-1]:
                node_representation = node_representation + h
        
        h = self.pool(node_representation, batch)
        h = self.feat_lin(h)
        output = self.pred_head(h)
        output = torch.sigmoid(output)
        
        return output
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
