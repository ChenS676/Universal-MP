
import os
import sys
import csv
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
import numpy as np
from yacs.config import CfgNode
from pprint import pprint 
from train_utils import train, valid, test

from baselines.utils import loaddataset

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from baselines.heuristic import AA, RA, Ben_PPR, katz_apro, shortest_path
from baselines.heuristic import CN as CommonNeighbor
from baselines.GNN import  LinkPredictor, GAE4LP
from yacs.config import CfgNode as CN
from archiv.mlp_heuristic_main import EarlyStopping
from baselines.LINKX import LINKX
from train_utils import  test
from utils import save_to_csv, visualize, set_random_seeds
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm 

from torch import Tensor
from torch_sparse import SparseTensor

ROOT = os.path.dirname(os.path.abspath(__file__))

class Config:
    def __init__(self):
        self.epochs = 100
        self.dataset = "ddi"
        self.batch_size = 8192
        self.h_key = "CN"
        self.gnn = "gcn"
        self.model = "LINKX"
        self.use_feature = False
        self.node_feature = 'original'  # 'one-hot', 'random', 
                                        # 'quasi-orthogonal', 'adjacency'
        self.use_early_stopping = True

        # optimizers
        self.lr = 0.01
        self.weight_decay = 0.0005


def spmdiff_efficient(adj1: SparseTensor, adj2: SparseTensor, keep_val: bool = False) -> SparseTensor:
    """
    Efficiently return the elements in adj1 but not in adj2.
    """
    # Ensure the input tensors have the same dimensions
    assert adj1.sizes() == adj2.sizes(), "Sparse tensors must have the same dimensions"
    
    row1, col1, val1 = adj1.coo()
    row2, col2, val2 = adj2.coo()

    keys1 = row1 * adj1.size(1) + col1  # Encode 2D indices into unique 1D keys
    keys2 = row2 * adj2.size(1) + col2

    mask1 = ~torch.isin(keys1, keys2)  # Mask elements in adj1 not found in adj2
    row_diff = row1[mask1]
    col_diff = col1[mask1]

    val1 = val1 if val1 is not None else torch.ones_like(row1, dtype=torch.float)
    val2 = val2 if val2 is not None else torch.ones_like(row2, dtype=torch.float)
            
    if keep_val:
        val_diff = val1[mask1]
        return SparseTensor.from_edge_index(
            torch.stack([row_diff, col_diff], dim=0),
            val_diff,
            adj1.sizes()
        )
    else:
        return SparseTensor.from_edge_index(
            torch.stack([row_diff, col_diff], dim=0),
            None,
            adj1.sizes()
        )

@torch.no_grad()
def valid_cn(encoder, predictor, data, splits, batch_size, device):
         
    encoder.eval()
    predictor.eval()

    pos_valid_edge = splits['pos_edge_label_index'].to(device)
    neg_valid_edge = splits['neg_edge_label_index'].to(device)
    pos_edge_label = splits['pos_edge_score'].to(device)
    neg_edge_label = splits['neg_edge_score'].to(device)


    for perm in tqdm(DataLoader(range(pos_valid_edge.size(0)), batch_size,
                           shuffle=True), desc='Valid/Test'):
        edge = pos_valid_edge[perm].t() 
        
        h = encoder(data.x, data.full_adj_t)
        neg_edge = neg_valid_edge[:, perm]
        
        pos_pred = predictor(h, edge).squeeze()
        neg_pred = predictor(h, neg_edge).squeeze()

        pos_loss = F.mse_loss(pos_pred, pos_edge_label[perm])
        neg_loss = F.mse_loss(neg_pred, neg_edge_label[perm])
        loss = pos_loss + neg_loss

    return loss.item()


def train_cn(encoder, predictor, optimizer, data, split_edge, batch_size, mask_target=True):
    encoder.train()
    predictor.train()
    
    device = data.adj_t.device()

    pos_train_edge = split_edge['train']['pos_edge_label_index'].to(device)
    neg_edge_epoch = split_edge['train']['neg_edge_label_index'].to(device)
    pos_edge_label = split_edge['train']['pos_edge_score'].to(device)
    neg_edge_label = split_edge['train']['neg_edge_score'].to(device)

    optimizer.zero_grad()

    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True), desc='Train'):
        edge = pos_train_edge[perm].t()
        if mask_target:
            
            adj_t = data.adj_t
            undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
            adj_t = spmdiff_efficient(adj_t, target_adj, keep_val=True)
        else:
            adj_t = data.adj_t

        h = encoder(data.x, adj_t)
        neg_edge = neg_edge_epoch[:,perm]
        
        pos_pred = predictor(h, edge).squeeze()
        neg_pred = predictor(h, neg_edge).squeeze()

        pos_loss = F.mse_loss(pos_pred, pos_edge_label[perm])
        neg_loss = F.mse_loss(neg_pred, neg_edge_label[perm])
        loss = pos_loss + neg_loss

        loss.backward()

        if data.x is not None:
            torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()


def experiment_loop(args: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and set up data
    data, splits = loaddataset(args.dataset, None)
    data = data.to(device)

    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )

    # Config node feature
    if args.node_feature == 'one-hot':
        data.x = torch.eye(data.num_nodes, data.num_nodes).to(device)
    elif args.node_feature == 'random':
        dim_feat = data.x.size(1)
        data.x = torch.randn(data.num_nodes, dim_feat).to(device)
    elif args.node_feature == 'adjacency':
        A_dense = A.toarray()
        A_tensor = torch.tensor(A_dense)
        data.x = A_tensor.float().to(device)
    elif args.node_feature == 'original':
        pass
    else:
        raise NotImplementedError(
            f'node_feature: {args.node_feature} is not implemented.'
        )

    # Initialize model configuration
    with open(ROOT + '/exp1_gnn.yaml', "r") as f:
        cfg = CfgNode.load_cfg(f)

    cfg_encoder = eval(f'cfg.model.{args.model}')
    cfg_decoder = eval(f'cfg.score.{args.model}')
    cfg.model.type = args.model
    
    pprint(cfg_encoder)
    pprint(cfg_decoder)

    if not hasattr(splits['train'], 'x') or splits['train'].x is None:
        cfg_encoder.in_channels = 1024
    else:
        cfg_encoder.in_channels = data.x.size(1)

    heuristic = {
        "CN": CommonNeighbor,
        "AA": AA,
        "RA": RA,
        "PPR": Ben_PPR,
        "katz": katz_apro,
        "shortest_path": shortest_path
    }

    # Heuristic scores
    for key in splits:
        pos_edge_score, _ = heuristic[args.h_key](
            A, splits[key]['pos_edge_label_index'], batch_size=args.batch_size
        )
        neg_edge_score, _ = heuristic[args.h_key](
            A, splits[key]['neg_edge_label_index'], batch_size=args.batch_size
        )
        splits[key]['pos_edge_score'] = torch.sigmoid(pos_edge_score)
        splits[key]['neg_edge_score'] = torch.sigmoid(neg_edge_score)

    # refine it as class and save into .npz
    # Store results of each experiment
    early_stopping = EarlyStopping(patience=5, verbose=True)

    if args.model == 'LINKX':
        cfg_encoder.in_channels = data.x.size(-1)
        cfg_encoder.num_nodes = data.num_nodes

        encoder = LINKX(
            num_nodes=cfg_encoder.num_nodes,
            in_channels=cfg_encoder.in_channels,
            hidden_channels=cfg_encoder.hidden_channels,
            out_channels=cfg_encoder.out_channels,
            num_layers=cfg_encoder.num_layers,
            num_edge_layers=cfg_encoder.num_edge_layers,
            num_node_layers=cfg_encoder.num_node_layers,
        )

        decoder = LinkPredictor(cfg_encoder.out_channels,
                    cfg_decoder.score_hidden_channels,
                    cfg_decoder.score_out_channels,
                    cfg_decoder.score_num_layers,
                    cfg_decoder.score_dropout)

    encoder.reset_parameters()
    decoder.reset_parameters()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        data.adj_t = SparseTensor.from_edge_index(splits['train'].edge_index)
        train_loss = train_cn(
            encoder, decoder, optimizer, data, splits, args.batch_size
        )

        # Validate every 20 epochs
        if epoch % 20 == 0:
            valid_loss = valid_cn(encoder, decoder, data, splits['valid'], args.batch_size, device)
            print(
                f'Epoch: {epoch:03d}, train loss: {train_loss:.4f}, '
                f'valid loss: {valid_loss:.4f}, '
                f'Cost Time: {time.time() - start:.4f}s')

            if args.use_early_stopping:
                early_stopping(valid_loss)
                if early_stopping.early_stop:
                    print("Training stopped early!")
                    break
                
    data.adj_t = data.full_adj_t
    test_loss = valid_cn(encoder, decoder, data, splits['test'], args.batch_size, device)

    # Save results to CSV with mean and variance
    save_to_csv(f'./results/LINKX2{args.h_key}_{args.dataset}.csv',
                args.model,
                args.node_feature,
                args.h_key,
                test_loss)
    


if __name__ == "__main__":

    args = Config()
    for model in ['LINKX']:
        args.model = model
        for node_feature in ['original', 'one-hot', 'random', 'adjacency']:
            args.node_feature = node_feature
            for seed in range(3):
                set_random_seeds(seed)
                experiment_loop(args)