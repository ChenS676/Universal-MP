# adopted from benchmarking/exist_setting_ddi: Run models on on ogbl-ddi under the existing setting.
import sys
sys.path.append("..") 

import torch
import numpy as np
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

dir_path = get_root_dir()
log_print		= get_logger('testrun', 'log', get_config_dir())


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,data_name):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [20, 50, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])
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



def train(model, score_func, train_pos, data, emb, optimizer, batch_size, pos_train_weight, data_name):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
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
        h = model(x, adj)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # if data_name == 'ogbl-ddi':
        row, col, _ = adj.coo()
        edge_index = torch.stack([col, row], dim=0)
        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                num_neg_samples=perm.size(0), method='dense')
        # else:
        #     # Just do some trivial random sampling.
        #     edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
        #                         device=h.device)
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
def test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size, data_name, use_valedges_as_input):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges

    if emb == None: x = data.x
    else: x = emb.weight
    
    h = model(x, data.adj_t.to(x.device))
    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)

    pos_train_pred = test_edge(score_func, train_val_edge, h, batch_size)

    neg_valid_pred = test_edge(score_func, neg_valid_edge, h, batch_size)

    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size)

    if use_valedges_as_input:
        print('use_val_in_edge')
        h = model(x, data.full_adj_t.to(x.device))

    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size)

    neg_test_pred = test_edge(score_func, neg_test_edge, h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, data_name)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]
    return result, score_emb



# def main(count, lr, l2, dropout):
def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-ddi')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=65536)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_steps', type=int, default=1)

    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=30,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
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
    parser.add_argument('--name_tag', type=str, default='ddi')
    # args.lr = lr
    # args.l2 = l2
    # args.dropout = dropout

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=9999)
    
    args = parser.parse_args()
    if args.debug == True:
        print('debug mode with runs 2 and epochs 3')
        args.runs = 2
        args.epochs = 7
        args.eval_steps = 1
        args.name_tag = args.name_tag + '_debug'

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # dataset = Planetoid('.', 'cora')
    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(get_root_dir(), "dataset", args.data_name))
    
    data = dataset[0]
    edge_index = data.edge_index
    emb = None
    node_num = data.num_nodes
    split_edge = dataset.get_edge_split()
                
    ############################ preprocess data node feat ##########################
    if hasattr(data, 'x'):
        if data.x != None:
            x = data.x
            x = x.to(device)
            input_channel = x.size(1)
            if args.cat_n2v_feat:
                print('cat n2v embedding!!')
                n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
                data.x = torch.cat((data.x, n2v_emb), dim=-1)
        else:
            emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
            input_channel = args.hidden_channels
    else:
        emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
        input_channel = args.hidden_channels
    
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
    ############ process splits ################
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)
    else:
        data.full_adj_t = data.adj_t
    if args.data_name != 'ogbl-citation2':
        pos_train_edge = split_edge['train']['edge']
        pos_valid_edge = split_edge['valid']['edge']
        neg_valid_edge = split_edge['valid']['edge_neg']
        pos_test_edge = split_edge['test']['edge']
        neg_test_edge = split_edge['test']['edge_neg']
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)
        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        neg_valid_edge = split_edge['valid']['target_node_neg'] 
        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        neg_test_edge = split_edge['test']['target_node_neg']
        
        
    data = data.to(device)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    # train_pos = data['train_pos'].to(x.device)
    # eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        'mrr_hit20':  Logger(args.runs),
        'mrr_hit50':  Logger(args.runs),
        'mrr_hit100':  Logger(args.runs),
    }
    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ddi':
        eval_metric = 'Hits@20'
    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'

    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        init_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)
        # save_valid = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'valid_output'+str(run)
        # save_valid = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'valid_output'+str(run)
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
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, pos_train_edge, data, emb, optimizer, args.batch_size, train_edge_weight, args.data_name)
            # print(model.convs[0].att_src[0][0][:10])
            
            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.batch_size, args.data_name, args.use_valedges_as_input)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

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
                
                if len(loggers['AUC'].results[run]) > 0:
                    r = torch.tensor(loggers['AUC'].results[run])
                    best_valid_auc = round(r[:, 1].max().item(), 4)
                    best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
                    
                    print('AUC')
                    log_print.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                f'best test: {100*best_test_auc:.2f}%')
                print('---')
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save: save_emb(score_emb, save_path)
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
    
    result_all_run = {}
    save_dict = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
            if key == eval_metric:
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean
            result_all_run[key] = [mean_list, var_list]
            save_dict[key] = test_res
        # print(best_metric_valid_str +' ' +best_auc_valid_str)
    mvari_str2csv(args.name_tag, save_dict, f'results/{args.data_name}_lm_mrr.csv')
    if args.runs == 1:
        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
    else:
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))



if __name__ == "__main__":

    main()

