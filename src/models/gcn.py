import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool

from src.models.base import BaseModel


num_atom_type = 119
num_chirality_tag = 4


class SimpleGCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCNConv, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        deg.scatter_add_(0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        x = self.linear(x)
        
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.view(-1, 1))
        
        return out


class GCN(BaseModel):
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
        drop_ratio: float = 0.0,
        feat_dim: int = 256,
    ):
        super(GCN, self).__init__()
        self.num_layer = num_layers
        self.emb_dim = hidden_feat * 2
        self.drop_ratio = drop_ratio
        self.pooling = pooling

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(self.emb_dim, self.emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        if pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool

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
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        node_representation = h_list[-1]
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
