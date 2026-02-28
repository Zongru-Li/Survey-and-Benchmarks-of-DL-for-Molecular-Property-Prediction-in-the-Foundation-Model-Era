import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree

from src.models.base import BaseModel


num_atom_type = 119
num_chirality_tag = 4
num_bond_type = 5
num_bond_direction = 3


class MPNEncoder(nn.Module):
    def __init__(self, hidden_size, depth, dropout=0.1, activation='ReLU', atom_messages=True):
        super(MPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.atom_messages = atom_messages
        
        self.dropout_layer = nn.Dropout(p=dropout)
        
        if activation == 'ReLU':
            self.act_func = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.act_func = nn.LeakyReLU()
        elif activation == 'PReLU':
            self.act_func = nn.PReLU()
        else:
            self.act_func = nn.ReLU()
        
        self.W_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, edge_index):
        row, col = edge_index
        
        message = x
        
        for d in range(self.depth):
            agg = torch.zeros_like(message)
            
            if self.atom_messages:
                agg.index_add_(0, row, message[col])
                deg = degree(row, message.size(0), dtype=message.dtype).clamp(min=1)
            else:
                agg.index_add_(0, col, message[row])
                deg = degree(col, message.size(0), dtype=message.dtype).clamp(min=1)
            
            message_nei = agg / deg.unsqueeze(1)
            
            message = self.W_h(message_nei)
            message = self.act_func(message)
            message = self.dropout_layer(message)
            
            message = message + x
        
        return message


class DualMPNEncoder(nn.Module):
    def __init__(self, hidden_size, depth, dropout=0.1, activation='ReLU'):
        super(DualMPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.atom_encoder = MPNEncoder(hidden_size, depth, dropout, activation, atom_messages=True)
        self.bond_encoder = MPNEncoder(hidden_size, depth, dropout, activation, atom_messages=False)
        
    def forward(self, x, edge_index):
        atom_output = self.atom_encoder(x, edge_index)
        bond_output = self.bond_encoder(x, edge_index)
        
        return atom_output, bond_output


class CDMVGNN(BaseModel):
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
        ffn_num_layers: int = 2,
        dist_coff: float = 0.1,
    ):
        super(CDMVGNN, self).__init__()
        self.hidden_size = hidden_feat * 2
        self.emb_dim = self.hidden_size
        self.drop_ratio = drop_ratio
        self.pooling = pooling
        self.dist_coff = dist_coff
        
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        self.encoder = DualMPNEncoder(
            hidden_size=self.emb_dim,
            depth=num_layers,
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
        
        self.mol_atom_ffn = self._create_ffn(self.emb_dim, feat_dim, out, ffn_num_layers, drop_ratio)
        self.mol_bond_ffn = self._create_ffn(self.emb_dim, feat_dim, out, ffn_num_layers, drop_ratio)
        
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
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        
        atom_output, bond_output = self.encoder(x, edge_index)
        
        mol_atom_output = self.pool(atom_output, batch)
        mol_bond_output = self.pool(bond_output, batch)
        
        if self.training:
            atom_ffn_output = self.mol_atom_ffn(mol_atom_output)
            bond_ffn_output = self.mol_bond_ffn(mol_bond_output)
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_ffn(mol_atom_output)
            bond_ffn_output = self.mol_bond_ffn(mol_bond_output)
            
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
