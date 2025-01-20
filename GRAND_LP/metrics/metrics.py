import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    
    results = {}
    
    valid_auc = round(valid_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    
    valid_ap = round(valid_ap, 4)
    
    results['AP'] = valid_ap


    return results

def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}

def eval_mrr_batch(y_pred_pos, y_pred_neg, batch_size=1024):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
        batch_size: size of each batch for processing
    '''
    # Ensure y_pred_pos is reshaped correctly
    y_pred_pos = y_pred_pos.view(-1, 1)
    
    hits1_list, hits3_list, hits10_list = [], [], []
    hits20_list, hits50_list, hits100_list = [], [], []
    mrr_list = []

    # Process in batches to avoid memory issues
    num_samples = y_pred_pos.size(0)
    for i in range(0, num_samples, batch_size):
        batch_pos = y_pred_pos[i:i + batch_size]
        batch_neg = y_pred_neg[i:i + batch_size]

        # Calculate optimistic and pessimistic ranks
        optimistic_rank = (batch_neg >= batch_pos).sum(dim=1)
        pessimistic_rank = (batch_neg > batch_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

        # Calculate hits and mrr for the batch
        hits1_list.append((ranking_list <= 1).to(torch.float))
        hits3_list.append((ranking_list <= 3).to(torch.float))
        hits10_list.append((ranking_list <= 10).to(torch.float))
        hits20_list.append((ranking_list <= 20).to(torch.float))
        hits50_list.append((ranking_list <= 50).to(torch.float))
        hits100_list.append((ranking_list <= 100).to(torch.float))
        mrr_list.append(1.0 / ranking_list.to(torch.float))

    # Concatenate results from all batches
    hits1_list = torch.cat(hits1_list, dim=0)
    hits3_list = torch.cat(hits3_list, dim=0)
    hits10_list = torch.cat(hits10_list, dim=0)
    hits20_list = torch.cat(hits20_list, dim=0)
    hits50_list = torch.cat(hits50_list, dim=0)
    hits100_list = torch.cat(hits100_list, dim=0)
    mrr_list = torch.cat(mrr_list, dim=0)

    return { 
        'hits@1_list': hits1_list,
        'hits@3_list': hits3_list,
        'hits@20_list': hits20_list,
        'hits@50_list': hits50_list,
        'hits@10_list': hits10_list,
        'hits@100_list': hits100_list,
        'mrr_list': mrr_list
    }

def evaluate_mrr(pos_val_pred, neg_val_pred, opt):
                 
    # neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    if opt['dataset'] == 'ogbl-ppa':
        mrr_output =  eval_mrr_batch(pos_val_pred, neg_val_pred, opt['batch_size'])
    else:
        mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)
        
    valid_mrr=mrr_output['mrr_list'].mean().item()
    valid_mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    valid_mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    valid_mrr_hit10 = mrr_output['hits@10_list'].mean().item()

    valid_mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    valid_mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    valid_mrr_hit100 = mrr_output['hits@100_list'].mean().item()

    valid_mrr = round(valid_mrr, 4)
    valid_mrr_hit1 = round(valid_mrr_hit1, 4)
    valid_mrr_hit3 = round(valid_mrr_hit3, 4)
    valid_mrr_hit10 = round(valid_mrr_hit10, 4)

    valid_mrr_hit20 = round(valid_mrr_hit20, 4)
    valid_mrr_hit50 = round(valid_mrr_hit50, 4)
    valid_mrr_hit100 = round(valid_mrr_hit100, 4)
    
    results = {}
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10

    results['MRR'] = valid_mrr

    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100

    
    return results

def acc(pos_pred, neg_pred):
    hard_thres = (max(torch.max(pos_pred).item(), torch.max(neg_pred).item()) + min(torch.min(pos_pred).item(), torch.min(neg_pred).item())) / 2

    # Initialize predictions with zeros and set ones where condition is met
    y_pred = torch.zeros_like(pos_pred)
    y_pred[pos_pred >= hard_thres] = 1

    # Do the same for negative predictions
    neg_y_pred = torch.ones_like(neg_pred)
    neg_y_pred[neg_pred <= hard_thres] = 0

    # Concatenate the positive and negative predictions
    y_pred = torch.cat([y_pred, neg_y_pred], dim=0)

    # Initialize ground truth labels
    pos_y = torch.ones_like(pos_pred)
    neg_y = torch.zeros_like(neg_pred)
    y = torch.cat([pos_y, neg_y], dim=0)
    y_logits = torch.cat([pos_pred, neg_pred], dim=0)
    # Calculate accuracy    
    return (y == y_pred).float().mean().item()