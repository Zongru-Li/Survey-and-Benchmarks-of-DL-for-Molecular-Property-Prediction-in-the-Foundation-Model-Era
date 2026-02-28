import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree

from src.models.base import BaseModel


num_atom_type = 119
num_chirality_tag = 4
num_bond_type = 5
num_bond_direction = 3


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        return self.W_o(output)


class GroverMPNEncoder(nn.Module):
    def __init__(self, hidden_size, depth, dropout=0.1, activation='ReLU'):
        super(GroverMPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'ReLU':
            self.act_func = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.act_func = nn.LeakyReLU()
        else:
            self.act_func = nn.ReLU()
        
        self.W_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        
        for _ in range(self.depth):
            agg = torch.zeros_like(x)
            agg.index_add_(0, col, x[row])
            
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg = deg.clamp(min=1)
            
            message = agg / deg.unsqueeze(1)
            message = self.W_h(message)
            message = self.act_func(message)
            message = self.dropout(message)
            
            x = x + message
        
        return x


class GroverTransEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout=0.1, activation='ReLU'):
        super(GroverTransEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.atom_encoder = GroverMPNEncoder(hidden_size, num_layers, dropout, activation)
        self.bond_encoder = GroverMPNEncoder(hidden_size, num_layers, dropout, activation)
        
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers * 2)
        ])
        
    def forward(self, x, edge_index, edge_attr, batch):
        atom_output = self.atom_encoder(x, edge_index, edge_attr)
        
        num_graphs = batch.max().item() + 1
        batch_output = []
        for i in range(num_graphs):
            mask = (batch == i)
            graph_atoms = atom_output[mask]
            if graph_atoms.size(0) > 0:
                graph_atoms = graph_atoms.unsqueeze(0)
                for j, (attn, ffn, ln1, ln2) in enumerate(
                    zip(self.attention_layers, self.ffn_layers, 
                        self.layer_norms[::2], self.layer_norms[1::2])
                ):
                    attn_out = attn(graph_atoms, graph_atoms, graph_atoms)
                    graph_atoms = ln1(graph_atoms + attn_out)
                    ffn_out = ffn(graph_atoms)
                    graph_atoms = ln2(graph_atoms + ffn_out)
                batch_output.append(graph_atoms.squeeze(0))
        
        if len(batch_output) > 0:
            return torch.cat(batch_output, dim=0)
        return atom_output


class GROVER(BaseModel):
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
        drop_ratio: float = 0.1,
        feat_dim: int = 256,
        num_heads: int = 4,
        ffn_num_layers: int = 2,
        dist_coff: float = 0.1,
    ):
        super(GROVER, self).__init__()
        self.hidden_size = hidden_feat * 2
        self.emb_dim = self.hidden_size
        self.drop_ratio = drop_ratio
        self.pooling = pooling
        self.dist_coff = dist_coff
        
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        self.edge_embedding1 = nn.Embedding(num_bond_type, self.emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, self.emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        
        self.encoder = GroverTransEncoder(
            hidden_size=self.emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=drop_ratio
        )
        
        if pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool
        
        self.atom_ffn = self._create_ffn(self.emb_dim, feat_dim, out, ffn_num_layers, drop_ratio)
        self.bond_ffn = self._create_ffn(self.emb_dim, feat_dim, out, ffn_num_layers, drop_ratio)
        
        self.classification = False
        if self.classification:
            self.sigmoid = nn.Sigmoid()
    
    def _create_ffn(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        layers = [nn.Dropout(dropout)]
        
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                ])
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_embed = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        else:
            edge_embed = None
        
        atom_output = self.encoder(x, edge_index, edge_embed, batch)
        
        mol_output = self.pool(atom_output, batch)
        
        if self.training:
            atom_ffn_output = self.atom_ffn(mol_output)
            bond_ffn_output = self.bond_ffn(mol_output)
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.atom_ffn(mol_output)
            bond_ffn_output = self.bond_ffn(mol_output)
            
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)
            
            output = (atom_ffn_output + bond_ffn_output) / 2
            output = torch.sigmoid(output)
        
        return output
    
    def get_loss_func(self):
        def loss_func(preds, targets, dist_coff=self.dist_coff):
            pred_loss = nn.MSELoss(reduction='none')
            
            if type(preds) is not tuple:
                return pred_loss(preds, targets).mean()
            
            dist_loss = nn.MSELoss(reduction='none')
            
            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            
            return (pred_loss1 + pred_loss2 + dist_coff * dist).mean()
        
        return loss_func
