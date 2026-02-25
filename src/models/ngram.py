import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, SumPooling, MaxPooling


class NGramConv(nn.Module):
    def __init__(self, in_feats, out_feats, use_bias=True):
        super(NGramConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=use_bias)

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata['h'] = feat
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))
            h = g.ndata['h_sum']
            return self.linear(h)


class NgramRF(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(NgramRF, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.ngram_size = 6
        
        self.input_proj = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.ngram_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.ngram_layers.append(NGramConv(hidden_feat, hidden_feat, use_bias=use_bias))
            self.batch_norms.append(nn.BatchNorm1d(hidden_feat))
        
        self.ngram_weights = nn.Parameter(torch.ones(self.ngram_size) / self.ngram_size)
        
        self.linear_1 = nn.Linear(hidden_feat, out_feat, bias=True)
        self.linear_2 = nn.Linear(out_feat, out, bias=True)
        
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, features):
        h = self.input_proj(features)
        
        ngram_outputs = []
        h_ngram = h
        for i in range(self.ngram_size):
            for j, layer in enumerate(self.ngram_layers):
                h_ngram = layer(g, h_ngram)
                h_ngram = self.batch_norms[j](h_ngram)
                h_ngram = F.relu(h_ngram)
            
            if self.pooling == 'avg':
                y = self.avgpool(g, h_ngram)
            elif self.pooling == 'max':
                y = self.maxpool(g, h_ngram)
            elif self.pooling == 'sum':
                y = self.sumpool(g, h_ngram)
            else:
                y = self.avgpool(g, h_ngram)
            
            ngram_outputs.append(y)
        
        weights = F.softmax(self.ngram_weights, dim=0)
        combined = sum(w * o for w, o in zip(weights, ngram_outputs))
        
        out = self.linear_1(combined)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = torch.sigmoid(out)
        
        return out
    
    def get_grad_norm_weights(self):
        return self.parameters()


class NgramXGB(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(NgramXGB, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.ngram_size = 6
        
        self.input_proj = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.ngram_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.ngram_layers.append(NGramConv(hidden_feat, hidden_feat, use_bias=use_bias))
            self.batch_norms.append(nn.BatchNorm1d(hidden_feat))
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_feat, hidden_feat // 2),
            nn.Tanh(),
            nn.Linear(hidden_feat // 2, 1)
        )
        
        self.linear_1 = nn.Linear(hidden_feat, out_feat, bias=True)
        self.linear_2 = nn.Linear(out_feat, out, bias=True)
        
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, features):
        h = self.input_proj(features)
        
        ngram_outputs = []
        h_ngram = h
        for i in range(self.ngram_size):
            for j, layer in enumerate(self.ngram_layers):
                h_ngram = layer(g, h_ngram)
                h_ngram = self.batch_norms[j](h_ngram)
                h_ngram = F.relu(h_ngram)
            
            if self.pooling == 'avg':
                y = self.avgpool(g, h_ngram)
            elif self.pooling == 'max':
                y = self.maxpool(g, h_ngram)
            elif self.pooling == 'sum':
                y = self.sumpool(g, h_ngram)
            else:
                y = self.avgpool(g, h_ngram)
            
            ngram_outputs.append(y)
        
        stacked = torch.stack(ngram_outputs, dim=1)
        attn_weights = F.softmax(self.attention(stacked), dim=1)
        combined = torch.sum(stacked * attn_weights, dim=1)
        
        out = self.linear_1(combined)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = torch.sigmoid(out)
        
        return out
    
    def get_grad_norm_weights(self):
        return self.parameters()
