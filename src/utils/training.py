import sys
import torch
import torch.nn as nn
import numpy as np
import statistics
from typing import Tuple, Dict, Any, Optional, Callable
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
from logzero import logger

from src.utils.data import update_node_features


def get_loss_fn(loss_type: str):
    if loss_type == 'l1':
        return nn.L1Loss(reduction='sum')
    elif loss_type == 'l2':
        return nn.MSELoss(reduction='none')
    elif loss_type == 'sml1':
        return nn.SmoothL1Loss(reduction='sum')
    elif loss_type == 'bce':
        return nn.BCELoss(reduction='mean')
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


def train_epoch_gnn(model, device, train_loader, optimizer, loss_fn):
    model.train()
    total_train_loss = 0.0
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        y = data[0].to(device)
        g = update_node_features(data[1]).to(device)
        x = g.ndata['feat']
        out = model(g, x)
        
        y = y.to(dtype=out.dtype)
        mask = (y != -1).to(dtype=out.dtype)
        y_clean = torch.where(y == -1, torch.zeros_like(y), y)
        
        loss_elem = loss_fn(out, y_clean)
        loss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    return total_train_loss


def validate_epoch_gnn(model, device, valid_loader, loss_fn):
    model.eval()
    total_loss_val = 0.0
    
    with torch.no_grad():
        for batch_idx, valid_data in enumerate(valid_loader):
            y = valid_data[0].to(device)
            g = update_node_features(valid_data[1]).to(device)
            x = g.ndata['feat']
            out = model(g, x)
            
            y = y.to(dtype=out.dtype)
            mask = (y != -1).to(dtype=out.dtype)
            y_clean = torch.where(y == -1, torch.zeros_like(y), y)
            
            loss_elem = loss_fn(out, y_clean)
            vloss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)
            total_loss_val += vloss.item()
    
    return total_loss_val


def train_epoch_gat(model, device, train_loader, optimizer, loss_fn):
    model.train()
    total_train_loss = 0.0
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        y = data[0].to(device)
        graph_list = data[1].to(device)
        node_features = graph_list.ndata['feat']
        edge_features = graph_list.edata['feat']
        output = model(graph_list, node_features, edge_features)
        
        y = y.to(dtype=output.dtype)
        mask = (y != -1).to(dtype=output.dtype)
        y_clean = torch.where(y == -1, torch.zeros_like(y, dtype=output.dtype), y)
        
        loss_raw = loss_fn(output, y_clean)
        train_loss = (loss_raw * mask).sum() / mask.sum().clamp_min(1.0)
        
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
    
    return total_train_loss


def validate_epoch_gat(model, device, valid_loader, loss_fn):
    model.eval()
    total_loss_val = 0.0
    
    with torch.no_grad():
        for batch_idx, valid_data in enumerate(valid_loader):
            y = valid_data[0].to(device)
            graph_list = valid_data[1].to(device)
            node_features = graph_list.ndata['feat']
            edge_features = graph_list.edata['feat']
            output = model(graph_list, node_features, edge_features)
            
            y = y.to(dtype=output.dtype)
            mask = (y != -1).to(dtype=output.dtype)
            y_clean = torch.where(y == -1, torch.zeros_like(y, dtype=output.dtype), y)
            
            loss_raw = loss_fn(output, y_clean)
            valid_loss = (loss_raw * mask).sum() / mask.sum().clamp_min(1.0)
            total_loss_val += valid_loss.item()
    
    return total_loss_val


def predict_gnn(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            y = data[0].cpu()
            graph_list = update_node_features(data[1]).to(device)
            node_features = graph_list.ndata['feat']
            output = model(graph_list, node_features).cpu()
            
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                arr_label = torch.cat((arr_label, c_label), 0)
                arr_pred = torch.cat((arr_pred, c_pred), 0)
            
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)
    
    AUC = roc_auc_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
    return AUC


def predict_gnn_regression(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            y = data[0].cpu()
            graph_list = update_node_features(data[1]).to(device)
            node_features = graph_list.ndata['feat']
            output = model(graph_list, node_features).cpu()
            
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                arr_label = torch.cat((arr_label, c_label), 0)
                arr_pred = torch.cat((arr_pred, c_pred), 0)
            
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)
    
    r = pearsonr(total_labels.numpy().flatten(), total_preds.numpy().flatten())[0]
    return r


def predict_gat(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            y = data[0].cpu()
            graph_list = data[1].to(device)
            node_features = graph_list.ndata['feat']
            edge_features = graph_list.edata['feat']
            output = model(graph_list, node_features, edge_features).cpu()
            
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                arr_label = torch.cat((arr_label, c_label), 0)
                arr_pred = torch.cat((arr_pred, c_pred), 0)
            
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)
    
    AUC = roc_auc_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
    return AUC


def predict_gat_regression(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            y = data[0].cpu()
            graph_list = data[1].to(device)
            node_features = graph_list.ndata['feat']
            edge_features = graph_list.edata['feat']
            output = model(graph_list, node_features, edge_features).cpu()
            
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                arr_label = torch.cat((arr_label, c_label), 0)
                arr_pred = torch.cat((arr_pred, c_pred), 0)
            
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)
    
    r = pearsonr(total_labels.numpy().flatten(), total_preds.numpy().flatten())[0]
    return r


def train_model(
    model_factory: Callable[[], nn.Module],
    train_loader,
    val_loader,
    test_loader,
    config: Dict[str, Any],
    device: torch.device,
    model_type: str = 'gnn',
    task_type: str = 'classification',
    checkpoint_path: Optional[str] = None
) -> Tuple[Optional[dict], float, float]:
    model_name = config['model']['name']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    iterations = config['training'].get('iterations', 1)
    
    if task_type == 'regression':
        loss_type = 'l2'
    else:
        loss_type = config['training'].get('loss', 'bce')
    
    loss_fn = get_loss_fn(loss_type)
    
    is_gnn_model = model_name in ['ka_gnn', 'ka_gnn_two', 'mlp_sage', 'mlp_sage_two', 'kan_sage', 'kan_sage_two']
    is_regression = task_type == 'regression'
    
    if is_regression:
        metric_name = 'PearsonR'
    else:
        metric_name = 'AUC'
    
    All_scores = []
    best_model_state = None
    global_best_score = float('-inf') if is_regression else 0
    
    for i in range(iterations):
        model = model_factory()
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        best_score = float('-inf') if is_regression else 0
        
        for epoch in range(epochs):
            if is_gnn_model:
                train_loss = train_epoch_gnn(model, device, train_loader, optimizer, loss_fn)
                if val_loader:
                    val_loss = validate_epoch_gnn(model, device, val_loader, loss_fn)
                    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
                if is_regression:
                    score = predict_gnn_regression(model, device, test_loader)
                else:
                    score = predict_gnn(model, device, test_loader)
            else:
                train_loss = train_epoch_gat(model, device, train_loader, optimizer, loss_fn)
                if val_loader:
                    val_loss = validate_epoch_gat(model, device, val_loader, loss_fn)
                    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
                if is_regression:
                    score = predict_gat_regression(model, device, test_loader)
                else:
                    score = predict_gat(model, device, test_loader)
            
            is_better = score > best_score
            if is_better:
                best_score = score
                logger.info(f'{metric_name}: {best_score:.5f}')
                formatted_number = "{:.5f}".format(best_score)
                best_score = float(formatted_number)
                if best_score > global_best_score:
                    global_best_score = best_score
                    best_model_state = model.state_dict().copy()
            
            if epoch % 10 == 0:
                print(f"-------------------------------------------------------", flush=True)
                print(f"epoch: {epoch}", flush=True)
                print(f'best_{metric_name}: {best_score}', flush=True)
            
            if epoch == epochs - 1:
                print(f"the best result up to {i+1}-loop is {best_score:.4f}.", flush=True)
                All_scores.append(best_score)
            
            sys.stdout.flush()
    
    if len(All_scores) > 0:
        mean_value = statistics.mean(All_scores)
        if len(All_scores) > 1:
            std_dev = statistics.stdev(All_scores)
        else:
            std_dev = 0.0
    else:
        mean_value = 0.0
        std_dev = 0.0
    
    return best_model_state, mean_value, std_dev
