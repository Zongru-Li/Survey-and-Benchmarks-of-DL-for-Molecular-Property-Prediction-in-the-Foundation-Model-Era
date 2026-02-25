import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import SumPooling, AvgPooling, MaxPooling

from src.models.base import BaseModel


class DMPNNEncoderLayer(nn.Module):
    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        hidden_dim: int = 300,
        depth: int = 3,
        bias: bool = False,
        dropout: float = 0.0,
        aggregation: str = "mean",
        aggregation_norm: int = 100,
    ):
        super(DMPNNEncoderLayer, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm

        self.w_i = nn.Linear(atom_fdim + bond_fdim, hidden_dim, bias=bias)
        self.w_h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.w_o = nn.Linear(atom_fdim + hidden_dim, hidden_dim, bias=bias)

    def forward(self, g, atom_feats, bond_feats):
        with g.local_scope():
            src, dst = g.edges()
            num_edges = g.num_edges()
            num_nodes = g.num_nodes()
            device = atom_feats.device

            f_bonds = torch.cat([atom_feats[src], bond_feats], dim=1)
            h_0 = F.relu(self.w_i(f_bonds))

            rev_map = {}
            for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
                rev_map[(s, d)] = i

            reverse_e = torch.zeros(num_edges, dtype=torch.long, device=device)
            for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
                if (d, s) in rev_map:
                    reverse_e[i] = rev_map[(d, s)]
                else:
                    reverse_e[i] = i

            h = h_0.clone()

            for _ in range(self.depth - 1):
                g.ndata["tmp"] = torch.zeros(num_nodes, self.hidden_dim, device=device)
                g.edata["h_edge"] = h

                g.send_and_recv(
                    g.edges(),
                    fn.copy_e("h_edge", "m"),
                    fn.sum("m", "tmp")
                )

                incoming_sum = g.ndata["tmp"][src]

                rev_h = h[reverse_e]
                messages = incoming_sum - rev_h

                h_new = F.relu(self.w_h(messages))
                h = self.dropout(h_new)

            g.ndata["atom_msg"] = torch.zeros(num_nodes, self.hidden_dim, device=device)
            g.edata["h_final"] = h
            g.send_and_recv(
                g.edges(),
                fn.copy_e("h_final", "m"),
                fn.sum("m", "atom_msg")
            )

            atom_neigh_h = g.ndata["atom_msg"]
            atom_input = torch.cat([atom_feats, atom_neigh_h], dim=1)
            atom_hidden = F.relu(self.w_o(atom_input))
            atom_hidden = self.dropout(atom_hidden)

            return atom_hidden


class DMPNN(BaseModel):
    def __init__(
        self,
        in_feat: int,
        hidden_feat: int,
        out_feat: int,
        out: int,
        grid_feat: int = 1,
        num_layers: int = 3,
        pooling: str = "avg",
        use_bias: bool = False,
        dropout: float = 0.0,
        bond_fdim: int = 21,
    ):
        super(DMPNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.bond_fdim = bond_fdim

        self.encoder = DMPNNEncoderLayer(
            atom_fdim=in_feat,
            bond_fdim=bond_fdim,
            hidden_dim=hidden_feat,
            depth=num_layers,
            bias=use_bias,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_feat, out_feat),
            nn.ReLU(),
            nn.Linear(out_feat, out),
        )

        if pooling == "avg":
            self.pool = AvgPooling()
        elif pooling == "max":
            self.pool = MaxPooling()
        else:
            self.pool = SumPooling()

    def forward(self, g, h):
        atom_feats = h
        bond_feats = g.edata.get("feat", torch.zeros(g.num_edges(), self.bond_fdim, device=h.device))

        atom_hidden = self.encoder(g, atom_feats, bond_feats)
        graph_repr = self.pool(g, atom_hidden)
        output = self.ffn(graph_repr)
        return output

    def get_grad_norm_weights(self):
        return self.parameters()
