import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import torch.nn as nn
import argparse
import scipy.sparse as ssp
from baselines.gnn_utils import get_root_dir, get_logger, get_config_dir, evaluate_hits, evaluate_mrr, evaluate_auc, Logger, init_seed, save_emb
from baselines.gnn_utils import GCN, GAT, SAGE, GIN, MF, DGCNN, GCN_seal, SAGE_seal, DecoupleSEAL, mlp_score
import torch_geometric.transforms as T
# from logger import Logger
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
import os
from graphgps.utility.utils import mvari_str2csv, random_sampling_ogb
from ogb_linkx import LINKX, LINKX_WL
import torch
import pandas as pd

DATASET_PATH = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/dataset'

dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    k_list = [20, 50, 100]
    result = {}
    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])
    return result
        

def train(args, model, score_func, split_edge, train_pos, data, emb, indices, optimizer, batch_size, pos_train_weight, data_name):
    model.train()
    score_func.train()
    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    x = emb.weight
    emb_update = 1
    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1,0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        if pos_train_weight != None:
            edge_weight_mask = pos_train_weight[mask]
            edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
        else:
            edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        ###################
        # print(adj)
        if args.wl_process == 'norm':
            x = torch.cat([emb.weight, data.x], dim=-1)
            h = model(x, data.adj_t, data.edge_weight)
        elif args.wl_process == 'unique':
            h = model(indices, x, adj)
        else:
            h = model(x, data.adj_t, data.edge_weight)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        row, col, _ = adj.coo()
        edge_index = torch.stack([col, row], dim=0)
        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                num_neg_samples=perm.size(0), method='dense')
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size, negative_data=None):
    pos_preds = []
    neg_preds = []
    if negative_data is not None:
        for perm in DataLoader(range(input_data.size(0)),  batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))
            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()
            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
    pos_preds = torch.cat(pos_preds, dim=0)
    return pos_preds, neg_preds


@torch.no_grad()
def test(args, 
         model, 
         score_func, 
         data, 
         evaluation_edges, 
         emb, 
         indices, 
         evaluator_hit, 
         evaluator_mrr, 
         batch_size, 
         data_name, 
         use_valedges_as_input, 
         save):
    model.eval()
    score_func.eval()
    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges
    if emb == None: x = data.x
    else: x = emb.weight
    if args.wl_process == 'norm':
        x = torch.cat([emb.weight, data.x], dim=-1)
    else:
        pass

    if args.wl_process == 'norm':
        x = torch.cat([emb.weight, data.x], dim=-1)
        h = model(x, data.adj_t, data.edge_weight).to(x.device)
    elif args.wl_process == 'unique':
        h = model(indices, x, data.adj_t).to(x.device)
    else:
        h = model(x, data.adj_t, data.edge_weight).to(x.device)
            
    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)
    pos_valid_pred, neg_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size, negative_data=neg_valid_edge)
    pos_test_pred, neg_test_pred = test_edge(score_func, pos_test_edge, h, batch_size, negative_data=neg_test_edge)
    # print(' test_pos test_neg', pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = pos_valid_pred
    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
    
    if save is True:
        save_edge_predictions(pos_valid_edge, pos_valid_pred, 1, "pos_valid_preds.csv")
        save_edge_predictions(neg_valid_edge.view(-1, 2), neg_valid_pred.flatten(), 0, "neg_valid_preds.csv")
        save_edge_predictions(pos_test_edge, pos_test_pred, 1, "pos_test_preds.csv")
        save_edge_predictions(neg_valid_edge.view(-1, 2), neg_test_pred.flatten(), 0, "neg_test_preds.csv")
    else:
        print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
        score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]
        return result, score_emb


def save_edge_predictions(edge_index, preds, label, filename):
    """
    Save edge predictions to a CSV file.

    Args:
        edge_index (Tensor): Shape [2, num_edges] — source and target nodes.
        preds (Tensor): Shape [num_edges] — predicted scores.
        label (int): 1 for positive, 0 for negative.
        filename (str): Path to save the CSV.
    """
    src = edge_index[:, 0].cpu().numpy()
    dst = edge_index[:, 1].cpu().numpy()
    preds = preds.detach().cpu().numpy()
    labels = torch.tensor([label] * len(preds))

    df = pd.DataFrame({'src': src, 'dst': dst, 'pred': preds, 'label': labels})
    df.to_csv(filename, index=False)

# def main(count, lr, l2, dropout):
def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ddi')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='LINKX')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    ### train setting
    parser.add_argument('--batch_size', type=int, default=65536)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=30,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed',  type=int, default=999)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='Hits@50')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)
    ######gat
    parser.add_argument('--gat_head', type=int, default=1)
    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--cat_wl_feat', default=False, action='store_true')
    parser.add_argument('--wl_process', type=str)
    
    ######debug 
    # parser.add_argument('--runs', type=int, default=2)
    # parser.add_argument('--epochs', type=int, default=7)
    args = parser.parse_args()
    # args.lr = lr
    # args.l2 = l2
    # args.dropout = dropout
    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print(args)
    init_seed(args.seed)
    args.name_tag = f"{args.data_name}_gnn_{args.gnn_model}_{args.score_model}_cat{args.cat_wl_feat}_wl{args.wl_process}"

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # dataset = Planetoid('.', 'cora')
    dataset = PygLinkPropPredDataset(name='ogbl-'+args.data_name)
    
    data = dataset[0]
    edge_index = data.edge_index
    emb = None
    node_num = data.num_nodes
    split_edge = dataset.get_edge_split()
    
    # Load WL_test
    emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
    input_channel = args.hidden_channels
            
    if args.cat_wl_feat:
        print('cat wl embedding!!')
        wl_emb = torch.load(
                os.path.join(
                    DATASET_PATH, 
                    'wl_label/'+ 
                    'ogbl-' + args.data_name+
                    '_wl_labels.pt'))

        if args.wl_process == 'norm':
            normalized_wl = (wl_emb.float() - wl_emb.float().min()) / (wl_emb.float().max() - wl_emb.float().min())
            normalized_wl = normalized_wl.unsqueeze(-1)
            if data.x is None:
                data.x = normalized_wl
            indices = None
            args.gnn_model = 'LINKX'
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
    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            edge_weight = data.edge_weight.to(torch.float)
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            train_edge_weight = split_edge['train']['weight'].to(device)
            train_edge_weight = train_edge_weight.to(torch.float)
        else:
            train_edge_weight = None
    else:
        train_edge_weight = None
    data = T.ToSparseTensor()(data)
    data = data.to(device)
    n_nodes = data.num_nodes
    data = data.to(device)
    
    if args.cat_wl_feat:
        if args.wl_process == 'norm':
            input_channel += 1
        elif args.wl_process == 'unique':
            pass
        else:
            raise NotImplementedError
        
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
    else:
        model = eval(args.gnn_model)(input_channel, 
                args.hidden_channels,
                args.hidden_channels, 
                args.num_layers, 
                args.dropout, 
                args.gin_mlp_layer, 
                args.gat_head, 
                node_num, 
                args.cat_node_feat_mf).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)
    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
    }

    if args.data_name =='ddi':
        eval_metric = 'Hits@20'
    pos_train_edge = split_edge['train']['edge']
    pos_valid_edge = split_edge['valid']['edge']
    pos_test_edge = split_edge['test']['edge']

    with open(f'{args.input_dir}/ogbl-{args.data_name}/heart_valid_{args.filename}', "rb") as f:
        neg_valid_edge = np.load(f)
        neg_valid_edge = torch.from_numpy(neg_valid_edge)
    with open(f'{args.input_dir}/ogbl-{args.data_name}/heart_test_{args.filename}', "rb") as f:
        neg_test_edge = np.load(f)
        neg_test_edge = torch.from_numpy(neg_test_edge)
    
    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    print('train val val_neg test test_neg: ', pos_train_edge.size(), pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size())
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2
    for run in range(args.runs):
        print('#################################          ', run, '          #################################')
        import wandb
        wandb.init(project=f"{args.data_name}_", name=f"{args.data_name}_{args.gnn_model}_{args.score_model}_{args.name_tag}")
        wandb.config.update(args)
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        init_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)
        if emb != None:
            torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        score_func.reset_parameters()
        if emb != None:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)
        best_valid = 0
        kill_cnt = 0
        step = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(args, model, score_func, split_edge, pos_train_edge, data, emb, indices, optimizer, args.batch_size, train_edge_weight, args.data_name)
            # print(model.convs[0].att_src[0][0][:10])
            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(args, 
                                               model, 
                                               score_func, 
                                               data, 
                                               evaluation_edges, 
                                               emb, 
                                               indices, 
                                               evaluator_hit, 
                                               evaluator_mrr, 
                                               args.test_batch_size, 
                                               args.data_name, 
                                               args.use_valedges_as_input, 
                                               False)
                
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
                
                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                print(eval_metric)
                log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')
                print('---')
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save: 
                        torch.save(model.state_dict(), save_path+'_model')
                        torch.save(optimizer.state_dict(),save_path+'_op')
                        torch.save(emb,save_path+'_emb')
                        torch.save(score_func.state_dict(), save_path+'_predictor')
                        save_emb(score_emb, save_path)
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                loggers[key].print_statistics( run)
                print('\n')
        wandb.finish()
    test(args, 
        model, 
        score_func, 
        data, 
        evaluation_edges, 
        emb, 
        indices, 
        evaluator_hit, 
        evaluator_mrr, 
        args.test_batch_size, 
        args.data_name, 
        args.use_valedges_as_input, 
        True)
    
    save_dict = {}
    result_all_run = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
            if key == eval_metric:
                best_metric_valid_str = best_metric
                # best_valid_mean_metric = best_valid_mean
            if key == 'AUC':
                best_auc_valid_str = best_metric
                # best_auc_metric = best_valid_mean
            result_all_run[key] = [mean_list, var_list]
            save_dict[key] = test_res
    print(f"now save {save_dict}")
    mvari_str2csv(args.name_tag, save_dict, f'results_ddi_gnn/{args.data_name}_lm_mrr.csv')

    if args.runs == 1:
        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
    else:
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))



if __name__ == "__main__":

    main()
    
# adopted from benchmarking/exist_setting_ddi: Run models on on ogbl-ddi under the existing setting.
