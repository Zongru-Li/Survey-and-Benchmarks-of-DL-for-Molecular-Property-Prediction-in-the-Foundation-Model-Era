import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import Set2Set, GlobalAttention
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian

from src.models.base import BaseModel


num_atom_type = 120
num_chirality_tag = 4
num_bond_type = 5
num_bond_direction = 3


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list = [-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                - 1 / 2
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self._curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def _b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self._b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KANChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = 'sym',
        bias: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = nn.ModuleList([
            KANLinear(in_channels, out_channels, grid_size, spline_order, 0.1, 1.0, 1.0)
            for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _norm(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: OptTensor,
        normalization: str,
        lambda_max: OptTensor = None,
        dtype: int = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, dtype, num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:
        edge_index, norm = self._norm(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        Tx_1 = x
        out = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class GraphKAN(BaseModel):
    def __init__(
        self,
        in_feat: int = 113,
        hidden_feat: int = 64,
        out_feat: int = 32,
        out: int = 1,
        grid_feat: int = 1,
        num_layers: int = 3,
        pooling: str = 'mean',
        use_bias: bool = False,
        drop_ratio: float = 0.3,
        K: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super(GraphKAN, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.pooling = pooling

        self.x_embedding1 = nn.Embedding(num_atom_type, hidden_feat)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, hidden_feat)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        hidden_dims = [hidden_feat]
        for i in range(num_layers - 1):
            hidden_dims.append(hidden_feat // (2 ** (i + 1)) if hidden_feat // (2 ** (i + 1)) >= 32 else 32)
        hidden_dims.append(out_feat)

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            K_i = min(K + i, 5)
            self.convs.append(
                KANChebConv(
                    hidden_dims[i], hidden_dims[i + 1], K=K_i,
                    grid_size=grid_size, spline_order=spline_order
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dims[i + 1]))

        if pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_dims[-1], 1))
        elif pooling.startswith("set2set"):
            set2set_iter = int(pooling[-1]) if pooling[-1].isdigit() else 2
            self.pool = Set2Set(hidden_dims[-1], set2set_iter)
        else:
            self.pool = global_mean_pool

        pool_dim = hidden_dims[-1] * 2 if pooling.startswith("set2set") else hidden_dims[-1]
        self.graph_pred_linear = KANLinear(pool_dim, out, grid_size=grid_size, spline_order=spline_order)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if x.dtype == torch.long:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1] if x.size(1) > 1 else torch.zeros_like(x[:, 0]))

        edge_weight = None
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_weight = edge_attr.float()
            elif edge_attr.size(1) == 1:
                edge_weight = edge_attr.squeeze(-1).float()

        for i, (conv, ln) in enumerate(zip(self.convs, self.layer_norms)):
            x = conv(x, edge_index, edge_weight)
            x = ln(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, self.drop_ratio, training=self.training)

        if self.pooling == "set2set" or self.pooling == "attention":
            pooled = self.pool(x, batch)
        else:
            pooled = self.pool(x, batch)

        output = self.graph_pred_linear(pooled)
        output = torch.sigmoid(output)

        return output
