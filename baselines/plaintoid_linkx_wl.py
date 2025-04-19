import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Cora, Pubmed, Citeseer, Photo, Computer

import torch
import argparse
import scipy.sparse as ssp
from baselines.gnn_utils import get_root_dir, get_logger, get_config_dir, evaluate_hits, evaluate_mrr, evaluate_auc, Logger, init_seed, save_emb
from baselines.gnn_utils import GCN, GAT, SAGE, GIN, MF, DGCNN, GCN_seal, SAGE_seal, DecoupleSEAL, mlp_score

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from ogb_linkx import LINKX, LINKX_WL
from torch_geometric.datasets import Planetoid 
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from ncnc.ogbdataset import loaddataset
from graphgps.utility.utils import mvari_str2csv
from typing import Dict 
import torch.nn.functional as F
import torch.nn as nn

dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())
DATASET_PATH = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/dataset'

    
def read_data(data_name, neg_mode):
    data_name = data_name
    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    for split in ['train', 'test', 'valid']:
        if neg_mode == 'equal':
            path = dir_path+'/dataset' + '/{}/{}_pos.txt'.format(data_name, split)
        elif neg_mode == 'all':
            path = dir_path+'/dataset' + '/{}/allneg/{}_pos.txt'.format(data_name, split)
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            node_set.add(sub)
            node_set.add(obj)
            if sub == obj:
                continue
            if split == 'train': 
                train_pos.append((sub, obj))
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)
    for split in ['test', 'valid']:
        if neg_mode == 'equal':
            path = dir_path+'/dataset' + '/{}/{}_neg.txt'.format(data_name, split)
        elif neg_mode == 'all':
            path = dir_path+'/dataset' + '/{}/allneg/{}_neg.txt'.format(data_name, split)
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            if split == 'valid': 
                valid_neg.append((sub, obj))
            if split == 'test': 
                test_neg.append((sub, obj))

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 
    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
    train_pos_tensor = torch.tensor(train_pos)
    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)
    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)
    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]
    feature_embeddings = torch.load(dir_path+'/dataset' + '/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']
    data = {}
    data['adj'] = adj
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg
    data['x'] = feature_embeddings
    return data

    
def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 3, 10, 20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 20, 50, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1, 3, 10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)
    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    return result
       
    
def train(model, score_func, train_pos, x, indices, optimizer, batch_size):
    model.train()
    score_func.train()
    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), 
                           batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1,0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        ###################
        # print(adj)
        # h = model( x, adj)
        if indices is not None:
            h = model(indices, x, adj)
        else:
            h = model(x, adj)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size):
    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
    pred_all = torch.cat(preds, dim=0)
    return pred_all


@torch.no_grad()
def test(model, score_func, data, x, indices, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()
    # adj_t = adj_t.transpose(1,0)
    adj = data['adj'].to(x.device)
    if indices is not None:
        h = model(indices, x, adj)
    else:
        h = model(x, adj)
            
    # h = model(indices, x, data['adj'].to(x.device))
    # h = model(x, data['adj'].to(x.device))
    # print(h[0][:10])
    x = h
    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size)
    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size)
    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size)
    pos_test_pred = test_edge(score_func, data['test_pos'], h, batch_size)
    neg_test_pred = test_edge(score_func, data['test_neg'], h, batch_size)
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]
    return result, score_emb



    
def data2dict(data, splits, data_name) -> dict:
    #TODO test with all ogbl-datasets, start with collab
    if data_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo']:
        datadict = {}
        datadict.update({'adj': data.adj_t})
        datadict.update({'train_pos': splits['train']['edge']})
        datadict.update({'train_neg': splits['train']['edge_neg']})
        datadict.update({'valid_pos': splits['valid']['edge']})
        datadict.update({'valid_neg': splits['valid']['edge_neg']})
        datadict.update({'test_pos': splits['test']['edge']})
        datadict.update({'test_neg': splits['test']['edge_neg']})   
        datadict.update({'train_val': torch.cat([splits['valid']['edge'], splits['train']['edge']])})
        datadict.update({'x': data.x}) 
    else:
        raise ValueError('data_name not supported')
    return datadict


def wl_one_hot(wl_emb):
    unique_ids, mapped_ids = wl_emb.unique(return_inverse=True)
    one_hot_features = F.one_hot(mapped_ids, num_classes=len(unique_ids)).float()
    return one_hot_features


def wl_Embedding(wl_emb):

    num_roles = wl_emb.max().item() + 1  # or count unique values
    embedding_dim = 64
    wl_embedding = nn.Embedding(num_roles, embedding_dim)
    node_features = wl_embedding(wl_emb)
    return node_features


def wl_unique_Embedding(wl_emb, emb_dim=16):
    unique_hashes, indices = torch.unique(wl_emb, return_inverse=True)
    embedding_layer = nn.Embedding(num_embeddings=len(unique_hashes), embedding_dim=emb_dim)
    node_embeddings = embedding_layer(indices)
    return node_embeddings, embedding_layer


def wl_normalize(wl_emb):
    return (wl_emb.float() - wl_emb.float().min()) / (wl_emb.float().max() - wl_emb.float().min())


import math
def random_fourier_features(x, D=16, sigma=1.0):
    """
    x: A torch tensor of normalized WL hash IDs.
    D: The output dimension.
    sigma: Bandwidth parameter for the Gaussian kernel.
    """
    # Random weights and biases
    W = torch.randn(D) / sigma
    b = 2 * math.pi * torch.rand(D)
    
    # Compute the random Fourier features
    z = torch.cos(x.unsqueeze(1) * W + b)
    z = (2.0 / D) ** 0.5 * z
    return z




def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='Cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='LINKX')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--name_tag', type=str, default='None', required=False)
    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=2**9)
    parser.add_argument('--dropout', type=float, default=0.0)
    ### train setting
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--gin_mlp_layer', type=int, default=2)
    parser.add_argument('--gat_head', type=int, default=1)
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    parser.add_argument('--cat_wl_feat', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--wl_process', type=str)
    args = parser.parse_args()
    if args.debug == True:
        print('debug mode with runs 2 and epochs 3')
        args.runs = 2
        args.epochs = 7
        args.eval_steps = 1
        args.name_tag = args.name_tag + '_debug'
        
    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print('cat_wl_feat: ', args.cat_wl_feat)    
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # dataset = Planetoid('.', 'cora')

    # data = dataset[0]# 
    # readdata = read_data(args.data_name, args.neg_mode)
    args.name_tag = f"{args.data_name}_gnn_{args.gnn_model}_{args.score_model}_run{args.runs}"

    load_data, splits = loaddataset(args.data_name, False, None) 
    data = data2dict(load_data, splits, args.data_name)
    del load_data, splits
    emb = None
    node_num = data['x'].size(0) 
    x = data['x']
    if args.cat_n2v_feat:
        print('cat n2v embedding!!')
        n2v_emb = torch.load(
                    DATASET_PATH, 
                    'dataset', 
                    args.data_name+'-n2v-embedding.pt')
        x = torch.cat((x, n2v_emb), dim=-1)
    if args.cat_wl_feat:
        print('cat wl embedding!!')
        wl_emb = torch.load(
                os.path.join(
                    DATASET_PATH, 
                    'wl_label/'+ 
                    args.data_name+
                    '_wl_labels.pt'))
        if args.wl_process == 'norm':
            normalized_wl = (wl_emb.float() - wl_emb.float().min()) / (wl_emb.float().max() - wl_emb.float().min())
            normalized_wl = normalized_wl.unsqueeze(-1)
            x = torch.cat([x, normalized_wl], dim=1)
            indices = None
        elif args.wl_process == 'unique': 
            args.gnn_model = 'LINKX_WL'
            # wl_emb = wl_emb.unsqueeze(-1)
            unique_hashes, indices = torch.unique(wl_emb, return_inverse=True)
            wl_emb = nn.Embedding(num_embeddings=len(unique_hashes), embedding_dim=16)
            # rff_features = random_fourier_features(normalized_wl, D=16)
            indices = indices.to(device)
    else:
        wl_emb = None
        indices = None
        unique_hashes = None
        
    x = x.to(device)
    train_pos = data['train_pos'].to(x.device)
    input_channel = x.size(1)
    n_nodes = x.size(0)
    if args.gnn_model == 'LINKX_WL':
        model = LINKX_WL(
        num_nodes=n_nodes,
        in_channels=input_channel,
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        wl_emb_dim = 16,
        num_wl = len(unique_hashes),
    ).to(device)
    elif args.gnn_model == 'LINKX':
        model = LINKX(
        num_nodes=n_nodes,
        in_channels=input_channel,
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    # TODO 
    else:
        model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                        args.hidden_channels, args.num_layers, args.dropout, 
                        mlp_layer=args.gin_mlp_layer, head=args.gat_head, 
                        node_num=node_num, cat_node_feat_mf=args.cat_node_feat_mf,  
                        data_name=args.data_name).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
   
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC': Logger(args.runs),
        'AP': Logger(args.runs)
    }

    # import itertools
    # hyperparams = {
    #     'batch_size': [2**9],
    #     'lr': [0.001], #0.01, 0.001, 0.0001
    # }

    # for batch_size, lr in itertools.product(hyperparams['batch_size'], hyperparams['lr']):
    # args.batch_size = 2**9
    # args.lr = 0.001
    args.name_tag = (
    f'model_{args.gnn_model}'
    f'score_{args.score_model}'
    f'{args.data_name}_'
    # f'batch_size{args.batch_size}_'
    # f'lr{args.lr}_'
    f'cat_wl_{args.cat_wl_feat}_'
    f'wl_{args.wl_process}_'
    )

    for run in range(args.runs):
        import wandb
        wandb.init(project=f"{args.data_name}_", name=f"{args.data_name}_{args.gnn_model}_{args.score_model}_{args.name_tag}")
        wandb.config.update(args)
        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        init_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + \
                    str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' \
                        + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) \
                        + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+ \
                        str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        model.reset_parameters()
        score_func.reset_parameters()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Name: {name}, Shape: {param.shape}")
        for name, param in score_func.named_parameters():
            if param.requires_grad:
                print(f"Name: {name}, Shape: {param.shape}")
        
        try:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(wl_emb.parameters()),lr=args.lr, weight_decay=args.l2) # 
        except:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        best_test = 0
        step = 0
        
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, train_pos, x, indices, optimizer, args.batch_size)
            # print(model.convs[0].att_src[0][0][:10])
            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(model, score_func, data, x, indices, evaluator_hit, evaluator_mrr, args.batch_size)
                                            
                for key, result in results_rank.items():
                    wandb.log({'train_loss': loss}, step = step)
                    loggers[key].add_result(run, result)
                    wandb.log({f"Metrics/{key}": result[-1]}, step=step)
                    step += 1
                    
                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        print(key)
                        train_hits, valid_hits, test_hits = result
                        log_print.info(
                            f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save:
                        save_emb(score_emb, save_path)
                else:
                    kill_cnt += 1
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        wandb.finish()
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    result_all_run = {}
    save_dict = {}
    for key in loggers.keys():
        print(key)
        best_metric, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics() 
        loggers[key].print_statistics()
        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean
        result_all_run[key] = [mean_list, var_list]
        save_dict[key] = test_res
        print(save_dict)
    print(best_metric_valid_str + ' ' + best_auc_valid_str)
    mvari_str2csv(args.name_tag, save_dict, f'results/{args.data_name}_lm_mrr.csv')
    return best_valid_mean_metric, best_auc_metric, result_all_run

    

if __name__ == "__main__":
    main()
