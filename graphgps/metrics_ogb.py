import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch


"""
Evaluation functions from OGB.
https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py
"""

def eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}


def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return {'ap': sum(ap_list) / len(ap_list)}


def eval_rmse(y_true, y_pred):
    '''
        compute RMSE score averaged across tasks
    '''
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(
            ((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return {'rmse': sum(rmse_list) / len(rmse_list)}


def eval_acc(y_true, y_pred):
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return {'acc': sum(acc_list) / len(acc_list)}


def eval_F1(seq_ref, seq_pred):
    # '''
    #     compute F1 score averaged over samples
    # '''

    precision_list = []
    recall_list = []
    f1_list = []

    for l, p in zip(seq_ref, seq_pred):
        label = set(l)
        prediction = set(p)
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {'precision': np.average(precision_list),
            'recall': np.average(recall_list),
            'F1': np.average(f1_list)}


# def evaluate_hits(test_pred, labels)
def evaluate_hits(evaluator, pos_pred, neg_pred, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        hits = round(hits, 4)

        results[f'Hits@{K}'] = hits

    return results
        

def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred):
                 
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    
    mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)

    valid_mrr =mrr_output['mrr_list'].mean().item()
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


def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    
    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)
    
    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)
    
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


def eval_hard_negs(pos_pred, neg_pred, k_list):
    """
    Eval on hard negatives
    """
    neg_pred = neg_pred.squeeze(-1)

    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (neg_pred >= pos_pred).sum(dim=-1)

    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (neg_pred > pos_pred).sum(dim=-1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    results = {}
    for k in k_list:
        mean_score = (ranking_list <= k).to(torch.float).mean().item()
        results[f'Hits@{k}'] = round(mean_score, 4)

    mean_mrr = 1./ranking_list.to(torch.float)
    results['MRR'] = round(mean_mrr.mean().item(), 4)

    return results


def get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred):

    k_list  = [1, 3, 10, 20, 50, 100]
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    result = {
        f'Hits@{K}': result_hit_test[f'Hits@{K}']
        for K in [1, 3, 10, 20, 50, 100]
    }
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))

    result['MRR'] = (result_mrr_test['MRR'])
    result['mrr_hit1']  = (result_mrr_test['mrr_hit1'])
    result['mrr_hit3']  = (result_mrr_test['mrr_hit3'])
    result['mrr_hit10']  = (result_mrr_test['mrr_hit10'])
    result['mrr_hit20']  = (result_mrr_test['mrr_hit20'])
    result['mrr_hit50']  = (result_mrr_test['mrr_hit50'])
    result['mrr_hit100']  = (result_mrr_test['mrr_hit100'])
    # print(result_mrr_test['mrr_hit100'])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_test = evaluate_auc(test_pred, test_true)

    result['AUC'] = (result_auc_test['AUC'])
    result['AP'] = (result_auc_test['AP'])
    result['ACC'] = 0
    return result