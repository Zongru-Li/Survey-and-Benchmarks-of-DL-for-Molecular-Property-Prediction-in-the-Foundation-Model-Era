import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import SumPooling, AvgPooling, MaxPooling

from src.models.base import BaseModel


class AttentiveGRU(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int, hidden_size: int, dropout: float = 0.0):
        super(AttentiveGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.project_edge = nn.Linear(edge_feat_size, hidden_size, bias=False)
        self.project_node = nn.Linear(node_feat_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_feats, node_hidden):
        with g.local_scope():
            g.ndata["h"] = node_hidden
            g.edata["edge_proj"] = self.project_edge(edge_feats)
            g.ndata["node_proj"] = self.project_node(node_feats)
            
            g.apply_edges(fn.u_add_e("node_proj", "edge_proj", "edge_msg"))
            g.edata["alpha"] = F.softmax(g.edata["edge_msg"], dim=1)
            
            g.update_all(fn.u_mul_e("h", "alpha", "m"), fn.sum("m", "h_new"))
            
            new_hidden = self.gru(g.ndata["h_new"], node_hidden)
            return new_hidden


class GlobalReadout(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super(GlobalReadout, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats):
        with g.local_scope():
            g.ndata["h"] = node_feats
            g.ndata["att"] = self.attention(node_feats)
            
            g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h_sum"))
            g.update_all(fn.copy_u("att", "a"), fn.sum("a", "att_sum"))
            
            graph_repr = g.ndata["h_sum"] / (g.ndata["att_sum"] + 1e-8)
            return graph_repr


class AttentiveFPLayer(nn.Module):
    def __init__(self, node_feat_size: int, edge_feat_size: int, hidden_size: int, dropout: float = 0.0):
        super(AttentiveFPLayer, self).__init__()
        self.gru = AttentiveGRU(node_feat_size, edge_feat_size, hidden_size, dropout)

    def forward(self, g, node_feats, edge_feats, node_hidden):
        return self.gru(g, node_feats, edge_feats, node_hidden)


class AttentiveFP(BaseModel):
    def __init__(
        self,
        in_feat: int,
        hidden_feat: int,
        out_feat: int,
        out: int,
        grid_feat: int = 1,
        num_layers: int = 2,
        num_timesteps: int = 2,
        pooling: str = "avg",
        use_bias: bool = False,
        dropout: float = 0.0,
        edge_feat_size: int = 21,
    ):
        super(AttentiveFP, self).__init__()
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.pooling = pooling
        self.edge_feat_size = edge_feat_size

        self.embedding = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        
        self.gnn_layers = nn.ModuleList([
            AttentiveFPLayer(hidden_feat, edge_feat_size, hidden_feat, dropout)
            for _ in range(num_layers)
        ])
        
        self.readout = GlobalReadout(hidden_feat, dropout)
        self.gru_readout = nn.GRU(hidden_feat, hidden_feat)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_feat, out_feat),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feat, out),
        )

        if pooling == "avg":
            self.pool = AvgPooling()
        elif pooling == "max":
            self.pool = MaxPooling()
        else:
            self.pool = SumPooling()

    def forward(self, g, h):
        node_feats = h
        edge_feats = g.edata.get("feat", torch.zeros(g.num_edges(), self.edge_feat_size, device=h.device))

        node_hidden = F.relu(self.embedding(node_feats))
        
        for gnn_layer in self.gnn_layers:
            node_hidden = gnn_layer(g, node_hidden, edge_feats, node_hidden)
        
        graph_hidden = self.pool(g, node_hidden)
        
        readout_hidden = graph_hidden.unsqueeze(0)
        for _ in range(self.num_timesteps):
            readout_hidden, _ = self.gru_readout(readout_hidden, readout_hidden)
        
        readout_hidden = readout_hidden.squeeze(0)
        output = self.predictor(readout_hidden)
        return output

    def get_grad_norm_weights(self):
        return self.parameters()
