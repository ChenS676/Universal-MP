

## debug on ppa  UPGRADE    
    - it never ends -> 1 epoch up to 2 hours why? with slightly improvement considering assing common neighbors

## debug on vessel  UPGRADE
    - [ ] no default parameters solved promptly, reference to statistic  -> use ddi 
    - [ ] function_laplacian_diffusion has OOM memory when running `ax = self.sparse_multiply(x)`
            - reimplement it using sparse tensor reduce epoch training time from 2.5 mins to 0.5min, but somehow some metrics have the decay.
- [ ] 5. citation2 UPGRADE
    - default parameter commented, no reason why
    - what to do -  uncomment

# Systematic Issue of High Complexity 

- [ ] measure the training and inference time of all gcns and save into txt
- [ ] measure the training and inference time of grands
- [ ] ### double check the evaluator of grand and the one from heart 
- [ ] provide table draft and fill the result in paper 

- [ ] Implement one random sampling method to deal with the long developement period 
- [x] Submit jobs to run ppa, vessel and citation2 for 4 epochs first, 
        - ppa epoch 1, 2, 3
        - vessel epoch 1, 2, 4
        - citation2 epoch 1, 2, 3

# How to choose hyperparameter for citation2, vessel 
    - I have an over-completed solution to calc the graph statistic with corresponding downsampling rate. 
    - While in the same time, just use parameter of ddi, it has until now the best performance.
    - for them we set the beltrami as False first #TODO beltrami true 
    - We indeed have no clue these statistic helps choosing a better hyper-parameters. 

# Report data statistic in paper draft
    Done: 
    Findings are: 
        - 1. citation network are more hierarchial Cora, Citesser
        - 2. ddi are very connected (function groups)
        - 3. vessel and citation2 are non-local, vessel is a mouse brain, citation2 is a collaboration network

## double check the evaluator of grand and the one from heart 

- [x]  first formatten the best parameters for all datasets 
- [ ]  generate nodevec for all listed dataset ogbl-ddi, ppa, vessel, citation2, collab, Cora, Citeseer, Pubmed 
- [ ]  create link
- [ ]  check the input and data dimension 
- [x]  TODO check the overlapping params betweena argparse and cfg_file for further analysis - use default value for ogbl-collab as test

- [x] 1. debug on ogbl-collab, 
    - download new version it is runnable. german has done a good job....
    - 
- [x] 2. debug on ogbl-ddi 
    - killed -> estimation is the hidden dimension is too high wrong
             -> check parameter integration method is only dopri5 for ddi  why? change back to dopri5
             -> PermIterator looks to be a bad dataloder
    - > identify as the overload of function torch.cat([]) -> change into list solve the problem 

- [x] 3. debug on ppa  UPGRADE
    - it never ends -> 1 epoch up to 2 hours why? with slightly improvement considering assing common neighbors 
    - slightly outperform

- [ ] 4. debug on vessel  UPGRADE
    - no default parameters solved promptly see below

- [ ] 5. citation2 UPGRADE
    - default parameter commented, no reason why 
    - what to do -  uncomment

Running Command:
```
python3 main_grand.py --dataset ogbl-ppa --device 0 --no_early --beltrami 
python3 main_grand.py --dataset ogbl-collab --device 0 --no_early --beltrami 
python3 main_grand.py --dataset ogbl-ddi --device 0 --no_early --beltrami  
# take very long several hours for one epoch
python main_grand.py  --dataset ogbl-vessel --device 0 --no_early --beltrami
```


Here is one important bug, check for all evaluator hitsk
    148 for K in [1, 3, 10, 20, 50, 100]:
    149     evaluator.K = K
--> 150     test_hits = evaluator.eval({
    151         'y_pred_pos': pos_pred,
    152         'y_pred_neg': neg_pred,
    153     })[f'hits@{K}']
    155 return test_hits, mrr, pos_pred, neg_pred

File ~/anaconda3/envs/EAsF/lib/python3.10/site-packages/ogb/linkproppred/evaluate.py:151, in Evaluator.eval(self, input_dict)
    148 def eval(self, input_dict):
    150     if 'hits@' in self.eval_metric:
--> 151         y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
    152         return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
    153     elif self.eval_metric == 'mrr':

File ~/anaconda3/envs/EAsF/lib/python3.10/site-packages/ogb/linkproppred/evaluate.py:84, in Evaluator._parse_and_check_input(self, input_dict)
     81         raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))
     83     if not y_pred_neg.ndim == 1:
---> 84         raise RuntimeError('y_pred_neg must to 1-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))
     86     return y_pred_pos, y_pred_neg, type_info
     88 elif 'mrr' == self.eval_metric:

RuntimeError: y_pred_neg must to 1-dim arrray, 2-dim array given