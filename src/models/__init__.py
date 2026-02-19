from src.models.base import BaseModel
from src.models.ka_gnn import KA_GNN, KA_GNN_two
from src.models.mlp_sage import MLPGNN, MLPGNN_two
from src.models.kan_sage import KANGNN, KANGNN_two
from src.models.ka_gat import KA_GAT
from src.models.mlp_gat import MLP_GAT
from src.models.kan_gat import KAN_GAT
from src.models.po_gat import PO_GAT


MODEL_REGISTRY = {
    'ka_gnn': KA_GNN,
    'ka_gnn_two': KA_GNN_two,
    'mlp_sage': MLPGNN,
    'mlp_sage_two': MLPGNN_two,
    'kan_sage': KANGNN,
    'kan_sage_two': KANGNN_two,
    'kagat': KA_GAT,
    'mlpgat': MLP_GAT,
    'kangat': KAN_GAT,
    'pogat': PO_GAT,
}


def get_model(config):
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    if model_name in ['ka_gnn', 'ka_gnn_two', 'mlp_sage', 'mlp_sage_two', 'kan_sage', 'kan_sage_two']:
        return model_class(
            in_feat=config['model']['in_feat'],
            hidden_feat=config['model']['hidden_feat'],
            out_feat=config['model']['out_feat'],
            out=config['model']['out_dim'],
            grid_feat=config['model'].get('grid_feat', 1),
            num_layers=config['model']['num_layers'],
            pooling=config['model']['pooling'],
            use_bias=config['model'].get('use_bias', False)
        )
    else:
        return model_class(
            in_node_dim=config['model']['in_node_dim'],
            in_edge_dim=config['model']['in_edge_dim'],
            hidden_dim=config['model']['hidden_feat'],
            out_1=config['model']['out_feat'],
            out_2=config['model']['out_dim'],
            gride_size=config['model'].get('grid_feat', 3),
            head=config['model'].get('num_heads', 2),
            layer_num=config['model']['num_layers'],
            pooling=config['model']['pooling']
        )
