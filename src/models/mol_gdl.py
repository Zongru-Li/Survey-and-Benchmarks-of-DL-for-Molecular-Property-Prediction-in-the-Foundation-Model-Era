import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, MaxPooling


class MultiScaleGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, use_bias=True):
        super(MultiScaleGCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=use_bias)

    def forward(self, g, feat):
        with g.local_scope():
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feat.device).unsqueeze(1)
            
            g.ndata['h'] = feat * norm
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = g.ndata['h'] * norm
            
            return self.linear(h)


class MolGDL(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(MolGDL, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.num_scales = 5
        self.hidden_feat = hidden_feat
        
        self.gcn_layers = nn.ModuleList()
        self.fc_projs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_scales):
            layers = nn.ModuleList()
            bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                layers.append(MultiScaleGCNConv(hidden_feat, hidden_feat, use_bias=use_bias))
                bns.append(nn.BatchNorm1d(hidden_feat))
            self.gcn_layers.append(layers)
            self.batch_norms.append(bns)
            self.fc_projs.append(nn.Linear(in_feat, hidden_feat, bias=use_bias))
        
        self.fc1 = nn.Linear(hidden_feat, hidden_feat)
        self.out = nn.Linear(hidden_feat, out)
        
        self.maxpool = MaxPooling()
        self.avgpool = AvgPooling()
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, features):
        scale_outputs = []
        
        for scale_idx in range(self.num_scales):
            h = self.fc_projs[scale_idx](features)
            
            for i, layer in enumerate(self.gcn_layers[scale_idx]):
                h = layer(g, h)
                h = self.batch_norms[scale_idx][i](h)
                h = F.relu(h)
            
            if self.pooling == 'max':
                y = self.maxpool(g, h)
            else:
                y = self.avgpool(g, h)
            
            scale_outputs.append(y)
        
        combined = torch.stack(scale_outputs, dim=0)
        h = torch.mean(combined, dim=0)
        
        h = F.normalize(h, p=2, dim=1)
        h = self.fc1(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)
        out = self.out(h)
        out = torch.sigmoid(out)
        
        return out
    
    def get_grad_norm_weights(self):
        return self.parameters()
